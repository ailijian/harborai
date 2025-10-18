#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态价格管理器测试

测试价格配置的热更新、审计日志、回滚功能和版本管理。
"""

import pytest
import asyncio
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from harborai.core.dynamic_pricing_manager import (
    DynamicPricingManager,
    PricingChangeRecord,
    PricingChangeType,
    PricingChangeStatus,
    PricingSnapshot,
    get_pricing_manager
)
from harborai.core.enhanced_pricing import EnhancedPricingCalculator


class TestDynamicPricingManager:
    """动态价格管理器测试类"""
    
    def setup_method(self):
        """测试前设置"""
        # 创建临时目录
        self.temp_dir = Path(tempfile.mkdtemp())
        self.audit_log_path = self.temp_dir / "pricing_audit.jsonl"
        self.snapshot_path = self.temp_dir / "snapshots"
        
        # 创建测试实例
        self.pricing_calculator = EnhancedPricingCalculator()
        self.manager = DynamicPricingManager(
            pricing_calculator=self.pricing_calculator,
            audit_log_path=str(self.audit_log_path),
            snapshot_path=str(self.snapshot_path)
        )
    
    def teardown_method(self):
        """测试后清理"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @pytest.mark.asyncio
    async def test_update_pricing_new_model(self):
        """测试新模型价格配置更新"""
        change_record = await self.manager.update_pricing(
            provider="openai",
            model="gpt-4-test",
            input_price_per_1k=0.03,
            output_price_per_1k=0.06,
            currency="USD",
            operator="test_user",
            reason="测试新模型价格"
        )
        
        assert change_record.change_type == PricingChangeType.CREATE
        assert change_record.status == PricingChangeStatus.APPLIED
        assert change_record.provider == "openai"
        assert change_record.model == "gpt-4-test"
        assert change_record.old_pricing is None
        assert change_record.new_pricing["input_price_per_1k"] == 0.03
        assert change_record.new_pricing["output_price_per_1k"] == 0.06
        assert change_record.operator == "test_user"
        
        # 验证价格配置已应用
        dynamic_key = "openai:gpt-4-test"
        assert dynamic_key in self.pricing_calculator.dynamic_pricing
        pricing = self.pricing_calculator.dynamic_pricing[dynamic_key]
        assert pricing.input_price_per_1k == 0.03
        assert pricing.output_price_per_1k == 0.06
        assert pricing.currency == "USD"
    
    @pytest.mark.asyncio
    async def test_update_pricing_existing_model(self):
        """测试现有模型价格配置更新"""
        # 先添加一个价格配置
        await self.manager.update_pricing(
            provider="openai",
            model="gpt-4-test",
            input_price_per_1k=0.03,
            output_price_per_1k=0.06,
            operator="test_user"
        )
        
        # 更新价格配置
        change_record = await self.manager.update_pricing(
            provider="openai",
            model="gpt-4-test",
            input_price_per_1k=0.025,
            output_price_per_1k=0.05,
            operator="test_user",
            reason="价格调整"
        )
        
        assert change_record.change_type == PricingChangeType.UPDATE
        assert change_record.status == PricingChangeStatus.APPLIED
        assert change_record.old_pricing["input_price_per_1k"] == 0.03
        assert change_record.new_pricing["input_price_per_1k"] == 0.025
        
        # 验证价格配置已更新
        dynamic_key = "openai:gpt-4-test"
        pricing = self.pricing_calculator.dynamic_pricing[dynamic_key]
        assert pricing.input_price_per_1k == 0.025
        assert pricing.output_price_per_1k == 0.05
    
    @pytest.mark.asyncio
    async def test_delete_pricing(self):
        """测试删除价格配置"""
        # 先添加一个价格配置
        await self.manager.update_pricing(
            provider="openai",
            model="gpt-4-test",
            input_price_per_1k=0.03,
            output_price_per_1k=0.06,
            operator="test_user"
        )
        
        # 删除价格配置
        change_record = await self.manager.delete_pricing(
            provider="openai",
            model="gpt-4-test",
            operator="test_user",
            reason="模型下线"
        )
        
        assert change_record.change_type == PricingChangeType.DELETE
        assert change_record.status == PricingChangeStatus.APPLIED
        assert change_record.old_pricing["input_price_per_1k"] == 0.03
        assert change_record.new_pricing is None
        
        # 验证价格配置已删除
        dynamic_key = "openai:gpt-4-test"
        assert dynamic_key not in self.pricing_calculator.dynamic_pricing
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent_pricing(self):
        """测试删除不存在的价格配置"""
        with pytest.raises(ValueError, match="价格配置不存在"):
            await self.manager.delete_pricing(
                provider="openai",
                model="nonexistent-model",
                operator="test_user"
            )
    
    @pytest.mark.asyncio
    async def test_batch_update_pricing(self):
        """测试批量更新价格配置"""
        pricing_updates = [
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
                "input_price_per_1k": 0.003,
                "output_price_per_1k": 0.006,
                "currency": "CNY"
            },
            {
                "provider": "baidu",
                "model": "ernie-bot",
                "input_price_per_1k": 0.008,
                "output_price_per_1k": 0.016
            }
        ]
        
        change_records = await self.manager.batch_update_pricing(
            pricing_updates=pricing_updates,
            operator="test_user",
            reason="批量价格初始化"
        )
        
        assert len(change_records) == 3
        
        for i, record in enumerate(change_records):
            assert record.change_type == PricingChangeType.CREATE
            assert record.status == PricingChangeStatus.APPLIED
            assert record.operator == "test_user"
            assert "批量更新" in record.reason
        
        # 验证所有价格配置都已应用
        assert len(self.pricing_calculator.dynamic_pricing) == 3
        assert "openai:gpt-4" in self.pricing_calculator.dynamic_pricing
        assert "deepseek:deepseek-chat" in self.pricing_calculator.dynamic_pricing
        assert "baidu:ernie-bot" in self.pricing_calculator.dynamic_pricing
    
    @pytest.mark.asyncio
    async def test_create_snapshot(self):
        """测试创建价格配置快照"""
        # 添加一些价格配置
        await self.manager.update_pricing(
            provider="openai",
            model="gpt-4",
            input_price_per_1k=0.03,
            output_price_per_1k=0.06,
            operator="test_user"
        )
        
        await self.manager.update_pricing(
            provider="deepseek",
            model="deepseek-chat",
            input_price_per_1k=0.003,
            output_price_per_1k=0.006,
            operator="test_user"
        )
        
        # 创建快照
        snapshot = await self.manager.create_snapshot(
            description="测试快照",
            operator="test_user"
        )
        
        assert snapshot.id.startswith("snapshot_")
        assert snapshot.description == "测试快照"
        assert len(snapshot.pricing_data) == 2
        assert "openai:gpt-4" in snapshot.pricing_data
        assert "deepseek:deepseek-chat" in snapshot.pricing_data
        assert snapshot.checksum is not None
        
        # 验证快照文件已创建
        snapshot_file = self.snapshot_path / f"{snapshot.id}.json"
        assert snapshot_file.exists()
    
    @pytest.mark.asyncio
    async def test_rollback_to_snapshot(self):
        """测试回滚到快照"""
        # 添加初始价格配置
        await self.manager.update_pricing(
            provider="openai",
            model="gpt-4",
            input_price_per_1k=0.03,
            output_price_per_1k=0.06,
            operator="test_user"
        )
        
        # 创建快照
        snapshot = await self.manager.create_snapshot(
            description="初始状态",
            operator="test_user"
        )
        
        # 修改价格配置
        await self.manager.update_pricing(
            provider="openai",
            model="gpt-4",
            input_price_per_1k=0.025,
            output_price_per_1k=0.05,
            operator="test_user"
        )
        
        # 添加新的价格配置
        await self.manager.update_pricing(
            provider="deepseek",
            model="deepseek-chat",
            input_price_per_1k=0.003,
            output_price_per_1k=0.006,
            operator="test_user"
        )
        
        # 验证当前状态
        assert len(self.pricing_calculator.dynamic_pricing) == 2
        gpt4_pricing = self.pricing_calculator.dynamic_pricing["openai:gpt-4"]
        assert gpt4_pricing.input_price_per_1k == 0.025
        
        # 回滚到快照
        rollback_records = await self.manager.rollback_to_snapshot(
            snapshot_id=snapshot.id,
            operator="test_user",
            reason="回滚测试"
        )
        
        assert len(rollback_records) == 1
        rollback_record = rollback_records[0]
        assert rollback_record.change_type == PricingChangeType.ROLLBACK
        assert rollback_record.provider == "openai"
        assert rollback_record.model == "gpt-4"
        
        # 验证回滚后状态
        assert len(self.pricing_calculator.dynamic_pricing) == 1
        assert "openai:gpt-4" in self.pricing_calculator.dynamic_pricing
        assert "deepseek:deepseek-chat" not in self.pricing_calculator.dynamic_pricing
        
        gpt4_pricing = self.pricing_calculator.dynamic_pricing["openai:gpt-4"]
        assert gpt4_pricing.input_price_per_1k == 0.03  # 回滚到原始值
        assert gpt4_pricing.output_price_per_1k == 0.06
    
    @pytest.mark.asyncio
    async def test_rollback_to_nonexistent_snapshot(self):
        """测试回滚到不存在的快照"""
        with pytest.raises(ValueError, match="快照.*不存在"):
            await self.manager.rollback_to_snapshot(
                snapshot_id="nonexistent_snapshot",
                operator="test_user"
            )
    
    @pytest.mark.asyncio
    async def test_pricing_validation(self):
        """测试价格配置验证"""
        # 测试负价格
        with pytest.raises(ValueError, match="输入价格.*超出允许范围"):
            await self.manager.update_pricing(
                provider="openai",
                model="gpt-4",
                input_price_per_1k=-0.01,
                output_price_per_1k=0.06,
                operator="test_user"
            )
        
        # 测试过高价格
        with pytest.raises(ValueError, match="输出价格.*超出允许范围"):
            await self.manager.update_pricing(
                provider="openai",
                model="gpt-4",
                input_price_per_1k=0.03,
                output_price_per_1k=1001.0,
                operator="test_user"
            )
        
        # 测试无效货币
        with pytest.raises(ValueError, match="货币.*不在允许列表中"):
            await self.manager.update_pricing(
                provider="openai",
                model="gpt-4",
                input_price_per_1k=0.03,
                output_price_per_1k=0.06,
                currency="INVALID",
                operator="test_user"
            )
    
    @pytest.mark.asyncio
    async def test_change_history(self):
        """测试变更历史查询"""
        # 添加一些变更记录
        await self.manager.update_pricing(
            provider="openai",
            model="gpt-4",
            input_price_per_1k=0.03,
            output_price_per_1k=0.06,
            operator="user1"
        )
        
        await self.manager.update_pricing(
            provider="deepseek",
            model="deepseek-chat",
            input_price_per_1k=0.003,
            output_price_per_1k=0.006,
            operator="user2"
        )
        
        await self.manager.update_pricing(
            provider="openai",
            model="gpt-4",
            input_price_per_1k=0.025,
            output_price_per_1k=0.05,
            operator="user1"
        )
        
        # 查询所有历史
        all_history = await self.manager.get_change_history()
        assert len(all_history) == 3
        
        # 查询特定厂商历史
        openai_history = await self.manager.get_change_history(provider="openai")
        assert len(openai_history) == 2
        assert all(record.provider == "openai" for record in openai_history)
        
        # 查询特定模型历史
        gpt4_history = await self.manager.get_change_history(provider="openai", model="gpt-4")
        assert len(gpt4_history) == 2
        assert all(record.model == "gpt-4" for record in gpt4_history)
        
        # 查询限制数量
        limited_history = await self.manager.get_change_history(limit=1)
        assert len(limited_history) == 1
    
    @pytest.mark.asyncio
    async def test_pricing_statistics(self):
        """测试价格配置统计信息"""
        # 添加一些变更记录
        await self.manager.update_pricing(
            provider="openai",
            model="gpt-4",
            input_price_per_1k=0.03,
            output_price_per_1k=0.06,
            operator="test_user"
        )
        
        await self.manager.update_pricing(
            provider="deepseek",
            model="deepseek-chat",
            input_price_per_1k=0.003,
            output_price_per_1k=0.006,
            operator="test_user"
        )
        
        await self.manager.delete_pricing(
            provider="deepseek",
            model="deepseek-chat",
            operator="test_user"
        )
        
        # 创建快照
        await self.manager.create_snapshot("测试快照", "test_user")
        
        # 获取统计信息
        stats = await self.manager.get_pricing_statistics()
        
        assert stats["total_changes"] == 3
        assert stats["successful_changes"] == 3
        assert stats["failed_changes"] == 0
        assert stats["success_rate"] == 1.0
        assert stats["change_types"]["create"] == 2
        assert stats["change_types"]["delete"] == 1
        assert stats["unique_providers"] == 2
        assert stats["unique_models"] == 2
        assert stats["total_snapshots"] == 1
        assert stats["current_dynamic_configs"] == 1
        assert stats["latest_change"] is not None
    
    @pytest.mark.asyncio
    async def test_audit_log_persistence(self):
        """测试审计日志持久化"""
        # 添加一些变更记录
        await self.manager.update_pricing(
            provider="openai",
            model="gpt-4",
            input_price_per_1k=0.03,
            output_price_per_1k=0.06,
            operator="test_user"
        )
        
        # 验证审计日志文件存在
        assert self.audit_log_path.exists()
        
        # 读取审计日志内容
        with open(self.audit_log_path, "r", encoding="utf-8") as f:
            log_lines = f.readlines()
        
        assert len(log_lines) == 1
        
        # 解析日志内容
        log_data = json.loads(log_lines[0].strip())
        assert log_data["change_type"] == "create"
        assert log_data["provider"] == "openai"
        assert log_data["model"] == "gpt-4"
        assert log_data["operator"] == "test_user"
        assert log_data["status"] == "applied"
        
        # 创建新的管理器实例，验证日志加载
        new_manager = DynamicPricingManager(
            pricing_calculator=EnhancedPricingCalculator(),
            audit_log_path=str(self.audit_log_path),
            snapshot_path=str(self.snapshot_path)
        )
        
        assert len(new_manager.change_records) == 1
        loaded_record = new_manager.change_records[0]
        assert loaded_record.provider == "openai"
        assert loaded_record.model == "gpt-4"
        assert loaded_record.operator == "test_user"
    
    @pytest.mark.asyncio
    async def test_snapshot_persistence(self):
        """测试快照持久化"""
        # 添加价格配置
        await self.manager.update_pricing(
            provider="openai",
            model="gpt-4",
            input_price_per_1k=0.03,
            output_price_per_1k=0.06,
            operator="test_user"
        )
        
        # 创建快照
        snapshot = await self.manager.create_snapshot("测试快照", "test_user")
        
        # 验证快照文件存在
        snapshot_file = self.snapshot_path / f"{snapshot.id}.json"
        assert snapshot_file.exists()
        
        # 读取快照内容
        with open(snapshot_file, "r", encoding="utf-8") as f:
            snapshot_data = json.load(f)
        
        assert snapshot_data["id"] == snapshot.id
        assert snapshot_data["description"] == "测试快照"
        assert "openai:gpt-4" in snapshot_data["pricing_data"]
        
        # 创建新的管理器实例，验证快照加载
        new_manager = DynamicPricingManager(
            pricing_calculator=EnhancedPricingCalculator(),
            audit_log_path=str(self.audit_log_path),
            snapshot_path=str(self.snapshot_path)
        )
        
        assert len(new_manager.snapshots) == 1
        loaded_snapshot = new_manager.snapshots[0]
        assert loaded_snapshot.id == snapshot.id
        assert loaded_snapshot.description == "测试快照"
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """测试错误处理"""
        # 模拟价格配置应用失败
        with patch.object(self.pricing_calculator, 'add_dynamic_pricing', side_effect=Exception("模拟错误")):
            with pytest.raises(Exception, match="模拟错误"):
                await self.manager.update_pricing(
                    provider="openai",
                    model="gpt-4",
                    input_price_per_1k=0.03,
                    output_price_per_1k=0.06,
                    operator="test_user"
                )
        
        # 验证失败记录被记录
        failed_records = [r for r in self.manager.change_records if r.status == PricingChangeStatus.FAILED]
        assert len(failed_records) == 1
        assert failed_records[0].error_message == "模拟错误"
    
    def test_pricing_change_record_checksum(self):
        """测试价格变更记录校验和"""
        record = PricingChangeRecord(
            id="test_id",
            change_type=PricingChangeType.CREATE,
            provider="openai",
            model="gpt-4",
            new_pricing={"input_price_per_1k": 0.03, "output_price_per_1k": 0.06},
            operator="test_user",
            reason="测试"
        )
        
        # 验证校验和生成
        assert record.checksum is not None
        assert len(record.checksum) == 32
        
        # 验证相同内容生成相同校验和
        record2 = PricingChangeRecord(
            id="test_id",
            change_type=PricingChangeType.CREATE,
            provider="openai",
            model="gpt-4",
            new_pricing={"input_price_per_1k": 0.03, "output_price_per_1k": 0.06},
            operator="test_user",
            reason="测试",
            timestamp=record.timestamp  # 使用相同时间戳
        )
        
        assert record.checksum == record2.checksum
    
    def test_pricing_snapshot_checksum(self):
        """测试价格配置快照校验和"""
        pricing_data = {
            "openai:gpt-4": {
                "input_price_per_1k": 0.03,
                "output_price_per_1k": 0.06,
                "currency": "USD"
            }
        }
        
        snapshot = PricingSnapshot(
            id="test_snapshot",
            timestamp=datetime.now(timezone.utc),
            pricing_data=pricing_data,
            version="v1",
            description="测试快照"
        )
        
        # 验证校验和生成
        assert snapshot.checksum is not None
        assert len(snapshot.checksum) == 64
        
        # 验证相同数据生成相同校验和
        snapshot2 = PricingSnapshot(
            id="test_snapshot2",
            timestamp=datetime.now(timezone.utc),
            pricing_data=pricing_data,
            version="v2",
            description="另一个测试快照"
        )
        
        assert snapshot.checksum == snapshot2.checksum  # 校验和只基于pricing_data


class TestGlobalPricingManager:
    """全局价格管理器测试类"""
    
    def test_get_pricing_manager_singleton(self):
        """测试全局价格管理器单例"""
        manager1 = get_pricing_manager()
        manager2 = get_pricing_manager()
        
        assert manager1 is manager2
        assert isinstance(manager1, DynamicPricingManager)
    
    def test_get_pricing_manager_with_custom_calculator(self):
        """测试使用自定义计算器的价格管理器"""
        custom_calculator = EnhancedPricingCalculator()
        manager = get_pricing_manager(custom_calculator)
        
        assert manager.pricing_calculator is custom_calculator


class TestIntegrationScenarios:
    """集成场景测试类"""
    
    def setup_method(self):
        """测试前设置"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.pricing_calculator = EnhancedPricingCalculator()
        self.manager = DynamicPricingManager(
            pricing_calculator=self.pricing_calculator,
            audit_log_path=str(self.temp_dir / "audit.jsonl"),
            snapshot_path=str(self.temp_dir / "snapshots")
        )
    
    def teardown_method(self):
        """测试后清理"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @pytest.mark.asyncio
    async def test_complete_pricing_lifecycle(self):
        """测试完整的价格配置生命周期"""
        # 1. 创建初始价格配置
        create_record = await self.manager.update_pricing(
            provider="openai",
            model="gpt-4",
            input_price_per_1k=0.03,
            output_price_per_1k=0.06,
            operator="admin",
            reason="初始配置"
        )
        assert create_record.change_type == PricingChangeType.CREATE
        
        # 2. 创建快照
        snapshot1 = await self.manager.create_snapshot("初始状态", "admin")
        
        # 3. 更新价格配置
        update_record = await self.manager.update_pricing(
            provider="openai",
            model="gpt-4",
            input_price_per_1k=0.025,
            output_price_per_1k=0.05,
            operator="admin",
            reason="价格调整"
        )
        assert update_record.change_type == PricingChangeType.UPDATE
        
        # 4. 创建另一个快照
        snapshot2 = await self.manager.create_snapshot("调整后状态", "admin")
        
        # 5. 回滚到初始状态
        rollback_records = await self.manager.rollback_to_snapshot(
            snapshot_id=snapshot1.id,
            operator="admin",
            reason="回滚到初始状态"
        )
        assert len(rollback_records) == 1
        assert rollback_records[0].change_type == PricingChangeType.ROLLBACK
        
        # 6. 验证最终状态
        final_pricing = self.pricing_calculator.dynamic_pricing["openai:gpt-4"]
        assert final_pricing.input_price_per_1k == 0.03  # 回滚到初始值
        assert final_pricing.output_price_per_1k == 0.06
        
        # 7. 验证历史记录
        history = await self.manager.get_change_history()
        assert len(history) == 3  # create, update, rollback
        
        # 8. 验证统计信息
        stats = await self.manager.get_pricing_statistics()
        assert stats["total_changes"] == 3
        assert stats["total_snapshots"] == 2
    
    @pytest.mark.asyncio
    async def test_concurrent_pricing_updates(self):
        """测试并发价格更新"""
        # 模拟并发更新不同模型
        tasks = []
        for i in range(5):
            task = self.manager.update_pricing(
                provider="test",
                model=f"model-{i}",
                input_price_per_1k=0.01 * (i + 1),
                output_price_per_1k=0.02 * (i + 1),
                operator=f"user-{i}",
                reason=f"并发测试 {i}"
            )
            tasks.append(task)
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks)
        
        # 验证所有更新都成功
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.status == PricingChangeStatus.APPLIED
            assert result.model == f"model-{i}"
        
        # 验证所有价格配置都已应用
        assert len(self.pricing_calculator.dynamic_pricing) == 5
        for i in range(5):
            key = f"test:model-{i}"
            assert key in self.pricing_calculator.dynamic_pricing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])