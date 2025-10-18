#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态价格管理器

支持价格配置的热更新、审计日志、回滚功能和版本管理。
实现完整的价格配置生命周期管理。
"""

import json
import asyncio
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum
import structlog

from .enhanced_pricing import EnhancedModelPricing, EnhancedPricingCalculator
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PricingChangeType(Enum):
    """价格变更类型"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    ROLLBACK = "rollback"
    BATCH_UPDATE = "batch_update"


class PricingChangeStatus(Enum):
    """价格变更状态"""
    PENDING = "pending"
    APPLIED = "applied"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class PricingChangeRecord:
    """价格变更记录"""
    id: str
    change_type: PricingChangeType
    provider: str
    model: str
    old_pricing: Optional[Dict[str, Any]] = None
    new_pricing: Optional[Dict[str, Any]] = None
    status: PricingChangeStatus = PricingChangeStatus.PENDING
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    operator: str = "system"
    reason: str = ""
    error_message: Optional[str] = None
    checksum: Optional[str] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = self._generate_id()
        if not self.checksum:
            self.checksum = self._calculate_checksum()
    
    def _generate_id(self) -> str:
        """生成变更记录ID"""
        timestamp_str = self.timestamp.strftime("%Y%m%d_%H%M%S_%f")
        content = f"{self.change_type.value}_{self.provider}_{self.model}_{timestamp_str}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _calculate_checksum(self) -> str:
        """计算变更记录校验和"""
        content = {
            "change_type": self.change_type.value,
            "provider": self.provider,
            "model": self.model,
            "old_pricing": self.old_pricing,
            "new_pricing": self.new_pricing,
            "timestamp": self.timestamp.isoformat(),
            "operator": self.operator,
            "reason": self.reason
        }
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()[:32]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "id": self.id,
            "change_type": self.change_type.value,
            "provider": self.provider,
            "model": self.model,
            "old_pricing": self.old_pricing,
            "new_pricing": self.new_pricing,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "operator": self.operator,
            "reason": self.reason,
            "error_message": self.error_message,
            "checksum": self.checksum
        }


@dataclass
class PricingSnapshot:
    """价格配置快照"""
    id: str
    timestamp: datetime
    pricing_data: Dict[str, Dict[str, Any]]
    version: str
    description: str = ""
    checksum: str = ""
    
    def __post_init__(self):
        if not self.checksum:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """计算快照校验和"""
        content_str = json.dumps(self.pricing_data, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "pricing_data": self.pricing_data,
            "version": self.version,
            "description": self.description,
            "checksum": self.checksum
        }


class DynamicPricingManager:
    """动态价格管理器
    
    功能特性：
    1. 价格配置热更新
    2. 完整的审计日志
    3. 版本管理和回滚
    4. 批量操作支持
    5. 配置验证和校验
    6. 自动备份和恢复
    """
    
    def __init__(self, 
                 pricing_calculator: EnhancedPricingCalculator,
                 audit_log_path: Optional[str] = None,
                 snapshot_path: Optional[str] = None):
        self.pricing_calculator = pricing_calculator
        self.logger = structlog.get_logger(__name__)
        
        # 配置文件路径
        self.audit_log_path = Path(audit_log_path or "logs/pricing_audit.jsonl")
        self.snapshot_path = Path(snapshot_path or "data/pricing_snapshots")
        
        # 确保目录存在
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_path.mkdir(parents=True, exist_ok=True)
        
        # 内存中的变更记录和快照
        self.change_records: List[PricingChangeRecord] = []
        self.snapshots: List[PricingSnapshot] = []
        
        # 配置验证规则
        self.validation_rules = {
            "min_price": 0.0,
            "max_price": 1000.0,
            "required_fields": ["input_price_per_1k", "output_price_per_1k"],
            "allowed_currencies": ["CNY", "USD", "EUR"]
        }
        
        # 加载历史记录
        self._load_audit_log()
        self._load_snapshots()
        
        self.logger.info("动态价格管理器初始化完成", 
                        audit_log_path=str(self.audit_log_path),
                        snapshot_path=str(self.snapshot_path))
    
    async def update_pricing(self,
                           provider: str,
                           model: str,
                           input_price_per_1k: float,
                           output_price_per_1k: float,
                           currency: str = "CNY",
                           operator: str = "system",
                           reason: str = "") -> PricingChangeRecord:
        """更新模型价格配置
        
        Args:
            provider: 厂商名称
            model: 模型名称
            input_price_per_1k: 每1K输入tokens的价格
            output_price_per_1k: 每1K输出tokens的价格
            currency: 货币单位
            operator: 操作者
            reason: 变更原因
            
        Returns:
            价格变更记录
        """
        try:
            # 验证价格配置
            await self._validate_pricing_config(input_price_per_1k, output_price_per_1k, currency)
            
            # 获取当前价格配置
            current_pricing = await self._get_current_pricing(provider, model)
            
            # 创建变更记录
            change_record = PricingChangeRecord(
                id="",  # 将在__post_init__中生成
                change_type=PricingChangeType.UPDATE if current_pricing else PricingChangeType.CREATE,
                provider=provider,
                model=model,
                old_pricing=current_pricing,
                new_pricing={
                    "input_price_per_1k": input_price_per_1k,
                    "output_price_per_1k": output_price_per_1k,
                    "currency": currency,
                    "source": "dynamic",
                    "last_updated": datetime.now(timezone.utc).isoformat()
                },
                operator=operator,
                reason=reason
            )
            
            # 应用价格变更
            self.pricing_calculator.add_dynamic_pricing(
                provider, model, input_price_per_1k, output_price_per_1k, currency
            )
            
            # 更新变更记录状态
            change_record.status = PricingChangeStatus.APPLIED
            
            # 记录审计日志
            await self._record_change(change_record)
            
            self.logger.info("价格配置更新成功",
                           provider=provider,
                           model=model,
                           change_id=change_record.id,
                           operator=operator)
            
            return change_record
            
        except Exception as e:
            # 创建失败记录
            change_record = PricingChangeRecord(
                id="",
                change_type=PricingChangeType.UPDATE,
                provider=provider,
                model=model,
                status=PricingChangeStatus.FAILED,
                operator=operator,
                reason=reason,
                error_message=str(e)
            )
            
            await self._record_change(change_record)
            
            self.logger.error("价格配置更新失败",
                            provider=provider,
                            model=model,
                            error=str(e),
                            operator=operator)
            
            raise
    
    async def delete_pricing(self,
                           provider: str,
                           model: str,
                           operator: str = "system",
                           reason: str = "") -> PricingChangeRecord:
        """删除模型价格配置
        
        Args:
            provider: 厂商名称
            model: 模型名称
            operator: 操作者
            reason: 删除原因
            
        Returns:
            价格变更记录
        """
        try:
            # 获取当前价格配置
            current_pricing = await self._get_current_pricing(provider, model)
            
            if not current_pricing:
                raise ValueError(f"模型 {provider}:{model} 的价格配置不存在")
            
            # 创建变更记录
            change_record = PricingChangeRecord(
                id="",
                change_type=PricingChangeType.DELETE,
                provider=provider,
                model=model,
                old_pricing=current_pricing,
                new_pricing=None,
                operator=operator,
                reason=reason
            )
            
            # 删除价格配置
            self.pricing_calculator.remove_dynamic_pricing(provider, model)
            
            # 更新变更记录状态
            change_record.status = PricingChangeStatus.APPLIED
            
            # 记录审计日志
            await self._record_change(change_record)
            
            self.logger.info("价格配置删除成功",
                           provider=provider,
                           model=model,
                           change_id=change_record.id,
                           operator=operator)
            
            return change_record
            
        except Exception as e:
            # 创建失败记录
            change_record = PricingChangeRecord(
                id="",
                change_type=PricingChangeType.DELETE,
                provider=provider,
                model=model,
                status=PricingChangeStatus.FAILED,
                operator=operator,
                reason=reason,
                error_message=str(e)
            )
            
            await self._record_change(change_record)
            
            self.logger.error("价格配置删除失败",
                            provider=provider,
                            model=model,
                            error=str(e),
                            operator=operator)
            
            raise
    
    async def batch_update_pricing(self,
                                 pricing_updates: List[Dict[str, Any]],
                                 operator: str = "system",
                                 reason: str = "") -> List[PricingChangeRecord]:
        """批量更新价格配置
        
        Args:
            pricing_updates: 价格更新列表，每项包含provider, model, input_price_per_1k, output_price_per_1k, currency
            operator: 操作者
            reason: 变更原因
            
        Returns:
            价格变更记录列表
        """
        change_records = []
        
        try:
            # 创建快照
            snapshot = await self.create_snapshot(f"批量更新前快照 - {operator}", operator)
            
            for update in pricing_updates:
                try:
                    change_record = await self.update_pricing(
                        provider=update["provider"],
                        model=update["model"],
                        input_price_per_1k=update["input_price_per_1k"],
                        output_price_per_1k=update["output_price_per_1k"],
                        currency=update.get("currency", "CNY"),
                        operator=operator,
                        reason=f"批量更新: {reason}"
                    )
                    change_records.append(change_record)
                    
                except Exception as e:
                    self.logger.error("批量更新中单项失败",
                                    provider=update.get("provider"),
                                    model=update.get("model"),
                                    error=str(e))
                    # 继续处理其他项目
                    continue
            
            self.logger.info("批量价格更新完成",
                           total_updates=len(pricing_updates),
                           successful_updates=len(change_records),
                           operator=operator)
            
            return change_records
            
        except Exception as e:
            self.logger.error("批量价格更新失败", error=str(e), operator=operator)
            raise
    
    async def rollback_to_snapshot(self,
                                 snapshot_id: str,
                                 operator: str = "system",
                                 reason: str = "") -> List[PricingChangeRecord]:
        """回滚到指定快照
        
        Args:
            snapshot_id: 快照ID
            operator: 操作者
            reason: 回滚原因
            
        Returns:
            回滚变更记录列表
        """
        try:
            # 查找快照
            snapshot = None
            for s in self.snapshots:
                if s.id == snapshot_id:
                    snapshot = s
                    break
            
            if not snapshot:
                raise ValueError(f"快照 {snapshot_id} 不存在")
            
            # 创建当前状态快照
            current_snapshot = await self.create_snapshot(f"回滚前快照 - {operator}", operator)
            
            # 清空当前动态价格配置
            current_dynamic_pricing = self.pricing_calculator.dynamic_pricing.copy()
            self.pricing_calculator.dynamic_pricing.clear()
            
            # 恢复快照中的价格配置
            change_records = []
            for key, pricing_data in snapshot.pricing_data.items():
                provider, model = key.split(":", 1)
                
                # 获取当前价格配置（转换为字典格式）
                old_pricing = None
                if key in current_dynamic_pricing:
                    old_pricing_obj = current_dynamic_pricing[key]
                    old_pricing = {
                        "input_price_per_1k": old_pricing_obj.input_price_per_1k,
                        "output_price_per_1k": old_pricing_obj.output_price_per_1k,
                        "currency": old_pricing_obj.currency,
                        "source": old_pricing_obj.source,
                        "last_updated": old_pricing_obj.last_updated.isoformat() if old_pricing_obj.last_updated else None
                    }
                
                change_record = PricingChangeRecord(
                    id="",
                    change_type=PricingChangeType.ROLLBACK,
                    provider=provider,
                    model=model,
                    old_pricing=old_pricing,
                    new_pricing=pricing_data,
                    status=PricingChangeStatus.APPLIED,
                    operator=operator,
                    reason=f"回滚到快照 {snapshot_id}: {reason}"
                )
                
                # 恢复价格配置
                self.pricing_calculator.add_dynamic_pricing(
                    provider=provider,
                    model=model,
                    input_price_per_1k=pricing_data["input_price_per_1k"],
                    output_price_per_1k=pricing_data["output_price_per_1k"],
                    currency=pricing_data.get("currency", "CNY")
                )
                
                change_records.append(change_record)
                await self._record_change(change_record)
            
            self.logger.info("快照回滚完成",
                           snapshot_id=snapshot_id,
                           restored_configs=len(change_records),
                           operator=operator)
            
            return change_records
            
        except Exception as e:
            self.logger.error("快照回滚失败",
                            snapshot_id=snapshot_id,
                            error=str(e),
                            operator=operator)
            raise
    
    async def create_snapshot(self,
                            description: str = "",
                            operator: str = "system") -> PricingSnapshot:
        """创建价格配置快照
        
        Args:
            description: 快照描述
            operator: 操作者
            
        Returns:
            价格配置快照
        """
        try:
            timestamp = datetime.now(timezone.utc)
            snapshot_id = f"snapshot_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}"
            
            # 获取当前所有动态价格配置
            pricing_data = {}
            for key, pricing in self.pricing_calculator.dynamic_pricing.items():
                pricing_data[key] = {
                    "input_price_per_1k": pricing.input_price_per_1k,
                    "output_price_per_1k": pricing.output_price_per_1k,
                    "currency": pricing.currency,
                    "source": pricing.source,
                    "last_updated": pricing.last_updated.isoformat() if pricing.last_updated else None
                }
            
            # 创建快照
            snapshot = PricingSnapshot(
                id=snapshot_id,
                timestamp=timestamp,
                pricing_data=pricing_data,
                version=f"v{len(self.snapshots) + 1}",
                description=description or f"自动快照 - {operator}"
            )
            
            # 保存快照
            await self._save_snapshot(snapshot)
            self.snapshots.append(snapshot)
            
            self.logger.info("价格配置快照创建成功",
                           snapshot_id=snapshot_id,
                           configs_count=len(pricing_data),
                           operator=operator)
            
            return snapshot
            
        except Exception as e:
            self.logger.error("价格配置快照创建失败", error=str(e), operator=operator)
            raise
    
    async def get_change_history(self,
                               provider: Optional[str] = None,
                               model: Optional[str] = None,
                               limit: int = 100) -> List[PricingChangeRecord]:
        """获取价格变更历史
        
        Args:
            provider: 厂商名称过滤
            model: 模型名称过滤
            limit: 返回记录数限制
            
        Returns:
            价格变更记录列表
        """
        filtered_records = []
        
        for record in reversed(self.change_records):  # 最新的在前
            if provider and record.provider != provider:
                continue
            if model and record.model != model:
                continue
            
            filtered_records.append(record)
            
            if len(filtered_records) >= limit:
                break
        
        return filtered_records
    
    async def get_pricing_statistics(self) -> Dict[str, Any]:
        """获取价格配置统计信息
        
        Returns:
            统计信息字典
        """
        total_changes = len(self.change_records)
        successful_changes = len([r for r in self.change_records if r.status == PricingChangeStatus.APPLIED])
        failed_changes = len([r for r in self.change_records if r.status == PricingChangeStatus.FAILED])
        
        change_types = {}
        for record in self.change_records:
            change_type = record.change_type.value
            change_types[change_type] = change_types.get(change_type, 0) + 1
        
        providers = set(record.provider for record in self.change_records)
        models = set(f"{record.provider}:{record.model}" for record in self.change_records)
        
        return {
            "total_changes": total_changes,
            "successful_changes": successful_changes,
            "failed_changes": failed_changes,
            "success_rate": successful_changes / total_changes if total_changes > 0 else 0,
            "change_types": change_types,
            "unique_providers": len(providers),
            "unique_models": len(models),
            "total_snapshots": len(self.snapshots),
            "current_dynamic_configs": len(self.pricing_calculator.dynamic_pricing),
            "latest_change": self.change_records[-1].timestamp.isoformat() if self.change_records else None
        }
    
    # ==================== 私有方法 ====================
    
    async def _validate_pricing_config(self,
                                     input_price: float,
                                     output_price: float,
                                     currency: str):
        """验证价格配置"""
        if input_price < self.validation_rules["min_price"] or input_price > self.validation_rules["max_price"]:
            raise ValueError(f"输入价格 {input_price} 超出允许范围 [{self.validation_rules['min_price']}, {self.validation_rules['max_price']}]")
        
        if output_price < self.validation_rules["min_price"] or output_price > self.validation_rules["max_price"]:
            raise ValueError(f"输出价格 {output_price} 超出允许范围 [{self.validation_rules['min_price']}, {self.validation_rules['max_price']}]")
        
        if currency not in self.validation_rules["allowed_currencies"]:
            raise ValueError(f"货币 {currency} 不在允许列表中: {self.validation_rules['allowed_currencies']}")
    
    async def _get_current_pricing(self, provider: str, model: str) -> Optional[Dict[str, Any]]:
        """获取当前价格配置"""
        dynamic_key = f"{provider}:{model}"
        if dynamic_key in self.pricing_calculator.dynamic_pricing:
            pricing = self.pricing_calculator.dynamic_pricing[dynamic_key]
            return {
                "input_price_per_1k": pricing.input_price_per_1k,
                "output_price_per_1k": pricing.output_price_per_1k,
                "currency": pricing.currency,
                "source": pricing.source,
                "last_updated": pricing.last_updated.isoformat() if pricing.last_updated else None
            }
        return None
    
    async def _record_change(self, change_record: PricingChangeRecord):
        """记录价格变更"""
        self.change_records.append(change_record)
        
        # 写入审计日志文件
        try:
            with open(self.audit_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(change_record.to_dict(), ensure_ascii=False) + "\n")
        except Exception as e:
            self.logger.error("审计日志写入失败", error=str(e))
    
    async def _save_snapshot(self, snapshot: PricingSnapshot):
        """保存快照到文件"""
        try:
            snapshot_file = self.snapshot_path / f"{snapshot.id}.json"
            with open(snapshot_file, "w", encoding="utf-8") as f:
                json.dump(snapshot.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error("快照保存失败", snapshot_id=snapshot.id, error=str(e))
            raise
    
    def _load_audit_log(self):
        """加载审计日志"""
        try:
            if self.audit_log_path.exists():
                with open(self.audit_log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            record_data = json.loads(line.strip())
                            record = PricingChangeRecord(
                                id=record_data["id"],
                                change_type=PricingChangeType(record_data["change_type"]),
                                provider=record_data["provider"],
                                model=record_data["model"],
                                old_pricing=record_data.get("old_pricing"),
                                new_pricing=record_data.get("new_pricing"),
                                status=PricingChangeStatus(record_data["status"]),
                                timestamp=datetime.fromisoformat(record_data["timestamp"]),
                                operator=record_data["operator"],
                                reason=record_data["reason"],
                                error_message=record_data.get("error_message"),
                                checksum=record_data.get("checksum")
                            )
                            self.change_records.append(record)
                
                self.logger.info("审计日志加载完成", records_count=len(self.change_records))
        except Exception as e:
            self.logger.error("审计日志加载失败", error=str(e))
    
    def _load_snapshots(self):
        """加载快照"""
        try:
            if self.snapshot_path.exists():
                for snapshot_file in self.snapshot_path.glob("*.json"):
                    with open(snapshot_file, "r", encoding="utf-8") as f:
                        snapshot_data = json.load(f)
                        snapshot = PricingSnapshot(
                            id=snapshot_data["id"],
                            timestamp=datetime.fromisoformat(snapshot_data["timestamp"]),
                            pricing_data=snapshot_data["pricing_data"],
                            version=snapshot_data["version"],
                            description=snapshot_data["description"],
                            checksum=snapshot_data["checksum"]
                        )
                        self.snapshots.append(snapshot)
                
                # 按时间排序
                self.snapshots.sort(key=lambda s: s.timestamp)
                
                self.logger.info("快照加载完成", snapshots_count=len(self.snapshots))
        except Exception as e:
            self.logger.error("快照加载失败", error=str(e))


# 全局动态价格管理器实例
_global_pricing_manager: Optional[DynamicPricingManager] = None


def get_pricing_manager(pricing_calculator: Optional[EnhancedPricingCalculator] = None) -> DynamicPricingManager:
    """获取全局动态价格管理器实例"""
    global _global_pricing_manager
    
    if _global_pricing_manager is None:
        if pricing_calculator is None:
            pricing_calculator = EnhancedPricingCalculator()
        _global_pricing_manager = DynamicPricingManager(pricing_calculator)
    
    return _global_pricing_manager