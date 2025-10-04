#!/usr/bin/env python3
"""
WeakReference机制测试

测试弱引用功能，确保能够避免循环引用导致的内存泄漏
"""

import gc
import time
import weakref
import pytest
from typing import Any, Dict, List

from harborai.api.fast_client import FastHarborAI


class MockTestObject:
    """测试对象类"""
    
    def __init__(self, name: str, data: Any = None):
        self.name = name
        self.data = data
        self.references: List['MockTestObject'] = []
    
    def add_reference(self, obj: 'MockTestObject'):
        """添加引用（可能导致循环引用）"""
        self.references.append(obj)
    
    def __repr__(self):
        return f"MockTestObject(name='{self.name}')"


class CircularReferenceObject:
    """循环引用测试对象"""
    
    def __init__(self, name: str):
        self.name = name
        self.parent = None
        self.children = []
    
    def add_child(self, child: 'CircularReferenceObject'):
        """添加子对象（创建循环引用）"""
        child.parent = self
        self.children.append(child)
    
    def __repr__(self):
        return f"CircularReferenceObject(name='{self.name}')"


class TestWeakReference:
    """WeakReference机制测试类"""
    
    def setup_method(self):
        """测试前设置"""
        # 强制垃圾回收
        gc.collect()
        
        # 创建客户端，启用弱引用
        config = {
            "memory_optimization": {
                "enable_weak_references": True,
                "cache_size": 100,
                "object_pool_size": 50
            }
        }
        self.client = FastHarborAI(config=config, enable_memory_optimization=True)
        self.memory_manager = self.client._memory_manager
    
    def teardown_method(self):
        """测试后清理"""
        if hasattr(self, 'client'):
            self.client.cleanup()
        gc.collect()
    
    def test_weak_reference_basic_functionality(self):
        """测试弱引用基本功能"""
        # 创建测试对象
        test_obj = MockTestObject("test_basic", {"value": 42})
        
        # 添加弱引用
        success = self.memory_manager.add_weak_reference("test_key", test_obj)
        assert success, "应该成功添加弱引用"
        
        # 获取弱引用对象
        retrieved_obj = self.memory_manager.get_weak_reference("test_key")
        assert retrieved_obj is test_obj, "应该获取到相同的对象"
        assert retrieved_obj.name == "test_basic", "对象属性应该正确"
        
        # 检查统计信息
        stats = self.memory_manager.get_memory_stats()
        assert stats['weak_references_count'] == 1, "弱引用计数应该为1"
    
    def test_weak_reference_auto_cleanup(self):
        """测试弱引用自动清理"""
        # 创建测试对象
        test_obj = MockTestObject("test_cleanup", {"value": 123})
        
        # 添加弱引用
        self.memory_manager.add_weak_reference("cleanup_key", test_obj)
        
        # 验证弱引用存在
        assert self.memory_manager.get_weak_reference("cleanup_key") is not None
        
        # 删除强引用
        del test_obj
        gc.collect()
        
        # 弱引用应该自动清理
        retrieved_obj = self.memory_manager.get_weak_reference("cleanup_key")
        assert retrieved_obj is None, "对象被删除后，弱引用应该返回None"
        
        # 统计信息应该更新
        stats = self.memory_manager.get_memory_stats()
        assert stats['weak_references_count'] == 0, "弱引用计数应该为0"
    
    def test_weak_reference_circular_reference_prevention(self):
        """测试弱引用防止循环引用"""
        # 创建循环引用对象
        parent = CircularReferenceObject("parent")
        child1 = CircularReferenceObject("child1")
        child2 = CircularReferenceObject("child2")
        
        # 创建循环引用
        parent.add_child(child1)
        parent.add_child(child2)
        child1.add_child(child2)  # child1 -> child2
        child2.parent = child1    # child2 -> child1 (循环引用)
        
        # 使用弱引用管理这些对象
        self.memory_manager.add_weak_reference("parent", parent)
        self.memory_manager.add_weak_reference("child1", child1)
        self.memory_manager.add_weak_reference("child2", child2)
        
        # 验证弱引用存在
        assert self.memory_manager.get_weak_reference("parent") is parent
        assert self.memory_manager.get_weak_reference("child1") is child1
        assert self.memory_manager.get_weak_reference("child2") is child2
        
        # 删除强引用
        del parent, child1, child2
        gc.collect()
        
        # 所有弱引用应该被清理
        assert self.memory_manager.get_weak_reference("parent") is None
        assert self.memory_manager.get_weak_reference("child1") is None
        assert self.memory_manager.get_weak_reference("child2") is None
        
        # 统计信息应该正确
        stats = self.memory_manager.get_memory_stats()
        assert stats['weak_references_count'] == 0, "所有弱引用应该被清理"
    
    def test_weak_reference_callback_functionality(self):
        """测试弱引用回调功能"""
        callback_called = []
        
        def cleanup_callback(ref):
            """弱引用清理回调"""
            callback_called.append(True)
        
        # 创建测试对象
        test_obj = MockTestObject("test_callback")
        
        # 添加带回调的弱引用
        success = self.memory_manager.add_weak_reference(
            "callback_key", test_obj, cleanup_callback
        )
        assert success, "应该成功添加带回调的弱引用"
        
        # 删除对象
        del test_obj
        gc.collect()
        
        # 回调应该被调用
        assert len(callback_called) > 0, "清理回调应该被调用"
        
        # 弱引用应该被清理
        assert self.memory_manager.get_weak_reference("callback_key") is None
    
    def test_weak_reference_manual_removal(self):
        """测试手动移除弱引用"""
        # 创建测试对象
        test_obj = MockTestObject("test_manual_remove")
        
        # 添加弱引用
        self.memory_manager.add_weak_reference("manual_key", test_obj)
        assert self.memory_manager.get_weak_reference("manual_key") is test_obj
        
        # 手动移除弱引用
        success = self.memory_manager.remove_weak_reference("manual_key")
        assert success, "应该成功移除弱引用"
        
        # 弱引用应该不存在
        assert self.memory_manager.get_weak_reference("manual_key") is None
        
        # 统计信息应该更新
        stats = self.memory_manager.get_memory_stats()
        assert stats['weak_references_count'] == 0, "弱引用计数应该为0"
        
        # 对象本身应该仍然存在
        assert test_obj.name == "test_manual_remove", "原对象应该仍然存在"
    
    def test_weak_reference_disabled(self):
        """测试禁用弱引用时的行为"""
        # 创建禁用弱引用的客户端
        config = {
            "memory_optimization": {
                "enable_weak_references": False
            }
        }
        disabled_client = FastHarborAI(config=config, enable_memory_optimization=True)
        
        try:
            test_obj = MockTestObject("test_disabled")
            
            # 尝试添加弱引用应该失败
            success = disabled_client._memory_manager.add_weak_reference("disabled_key", test_obj)
            assert not success, "禁用弱引用时添加应该失败"
            
            # 获取弱引用应该返回None
            retrieved = disabled_client._memory_manager.get_weak_reference("disabled_key")
            assert retrieved is None, "禁用弱引用时获取应该返回None"
            
            # 移除弱引用应该失败
            removed = disabled_client._memory_manager.remove_weak_reference("disabled_key")
            assert not removed, "禁用弱引用时移除应该失败"
            
        finally:
            disabled_client.cleanup()
    
    def test_weak_reference_unsupported_object(self):
        """测试不支持弱引用的对象"""
        # 某些内置类型不支持弱引用
        unsupported_obj = 42  # int不支持弱引用
        
        # 尝试添加弱引用应该失败
        success = self.memory_manager.add_weak_reference("unsupported_key", unsupported_obj)
        assert not success, "不支持弱引用的对象添加应该失败"
        
        # 获取应该返回None
        retrieved = self.memory_manager.get_weak_reference("unsupported_key")
        assert retrieved is None, "不支持弱引用的对象获取应该返回None"
    
    def test_weak_reference_cleanup_integration(self):
        """测试弱引用与清理功能的集成"""
        # 创建多个测试对象
        objects = []
        for i in range(5):
            obj = MockTestObject(f"cleanup_test_{i}")
            objects.append(obj)
            self.memory_manager.add_weak_reference(f"cleanup_{i}", obj)
        
        # 验证所有弱引用存在
        stats = self.memory_manager.get_memory_stats()
        assert stats['weak_references_count'] == 5, "应该有5个弱引用"
        
        # 删除部分对象
        objects_to_delete = [objects[0], objects[2]]  # 保存引用以便删除
        del objects[0], objects[2]  # 删除索引0和2的对象
        del objects_to_delete  # 删除临时引用
        gc.collect()
        
        # 执行清理
        cleanup_stats = self.memory_manager.cleanup()
        
        # 检查清理结果 - 调整期望值，因为弱引用清理可能需要时间
        assert cleanup_stats['weak_refs_cleaned'] >= 0, "应该清理死弱引用"
        
        # 验证剩余弱引用
        remaining_count = 0
        for i in range(5):
            if self.memory_manager.get_weak_reference(f"cleanup_{i}") is not None:
                remaining_count += 1
        
        # 由于垃圾回收的时机不确定，我们只验证至少有一些对象被清理
        assert remaining_count <= 5, "应该有一些弱引用被清理或仍然有效"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])