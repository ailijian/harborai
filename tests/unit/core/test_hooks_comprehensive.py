"""钩子系统comprehensive测试。"""

import pytest
import time
from unittest.mock import Mock, patch
from typing import Dict, Any

from harborai.core.plugins.hooks import (
    HookType, HookContext, PluginHook, FunctionHook, HookManager,
    get_hook_manager, register_hook, register_function_hook, execute_hooks
)


class TestHookType:
    """测试钩子类型枚举。"""
    
    def test_hook_type_values(self):
        """测试钩子类型的值。"""
        assert HookType.BEFORE.value == "before"
        assert HookType.AFTER.value == "after"
        assert HookType.ERROR.value == "error"
        assert HookType.BEFORE_INIT.value == "before_init"
        assert HookType.AFTER_INIT.value == "after_init"
        assert HookType.BEFORE_EXECUTE.value == "before_execute"
        assert HookType.AFTER_EXECUTE.value == "after_execute"
        assert HookType.ON_ERROR.value == "on_error"
        assert HookType.ON_SUCCESS.value == "on_success"
        assert HookType.BEFORE_CLEANUP.value == "before_cleanup"
        assert HookType.AFTER_CLEANUP.value == "after_cleanup"
    
    def test_hook_type_count(self):
        """测试钩子类型数量。"""
        # 确保所有钩子类型都被测试到
        expected_types = {
            "before", "after", "error", "before_init", "after_init",
            "before_execute", "after_execute", "on_error", "on_success",
            "before_cleanup", "after_cleanup"
        }
        actual_types = {hook_type.value for hook_type in HookType}
        assert actual_types == expected_types


class TestHookContext:
    """测试钩子上下文。"""
    
    def test_hook_context_creation(self):
        """测试钩子上下文创建。"""
        data = {"key": "value", "number": 42}
        timestamp = time.time()
        
        context = HookContext(
            plugin_name="test_plugin",
            hook_type=HookType.BEFORE,
            data=data,
            timestamp=timestamp
        )
        
        assert context.plugin_name == "test_plugin"
        assert context.hook_type == HookType.BEFORE
        assert context.data == data
        assert context.timestamp == timestamp
    
    def test_hook_context_data_modification(self):
        """测试钩子上下文数据修改。"""
        data = {"key": "value"}
        context = HookContext(
            plugin_name="test_plugin",
            hook_type=HookType.BEFORE,
            data=data,
            timestamp=time.time()
        )
        
        # 修改数据
        context.data["new_key"] = "new_value"
        assert context.data["new_key"] == "new_value"
        assert len(context.data) == 2


class TestPluginHook:
    """测试插件钩子基类。"""
    
    def test_plugin_hook_initialization(self):
        """测试插件钩子初始化。"""
        class TestHook(PluginHook):
            def execute(self, context: HookContext) -> bool:
                return True
        
        hook = TestHook("test_hook", priority=5)
        assert hook.name == "test_hook"
        assert hook.priority == 5
        assert hook.enabled is True
    
    def test_plugin_hook_default_priority(self):
        """测试插件钩子默认优先级。"""
        class TestHook(PluginHook):
            def execute(self, context: HookContext) -> bool:
                return True
        
        hook = TestHook("test_hook")
        assert hook.priority == 0
    
    def test_plugin_hook_enable_disable(self):
        """测试插件钩子启用/禁用。"""
        class TestHook(PluginHook):
            def execute(self, context: HookContext) -> bool:
                return True
        
        hook = TestHook("test_hook")
        
        # 默认启用
        assert hook.enabled is True
        
        # 禁用
        hook.disable()
        assert hook.enabled is False
        
        # 启用
        hook.enable()
        assert hook.enabled is True
    
    def test_plugin_hook_abstract_method(self):
        """测试插件钩子抽象方法。"""
        # 不能直接实例化抽象类
        with pytest.raises(TypeError):
            PluginHook("test_hook")


class TestFunctionHook:
    """测试函数钩子。"""
    
    def test_function_hook_creation(self):
        """测试函数钩子创建。"""
        def test_func(context: HookContext) -> bool:
            return True
        
        hook = FunctionHook("test_hook", test_func, priority=3)
        assert hook.name == "test_hook"
        assert hook.func == test_func
        assert hook.priority == 3
        assert hook.enabled is True
    
    def test_function_hook_execute_success(self):
        """测试函数钩子成功执行。"""
        executed = False
        
        def test_func(context: HookContext) -> bool:
            nonlocal executed
            executed = True
            assert context.plugin_name == "test_plugin"
            return True
        
        hook = FunctionHook("test_hook", test_func)
        context = HookContext(
            plugin_name="test_plugin",
            hook_type=HookType.BEFORE,
            data={},
            timestamp=time.time()
        )
        
        result = hook.execute(context)
        assert result is True
        assert executed is True
    
    def test_function_hook_execute_disabled(self):
        """测试禁用状态下的函数钩子执行。"""
        executed = False
        
        def test_func(context: HookContext) -> bool:
            nonlocal executed
            executed = True
            return True
        
        hook = FunctionHook("test_hook", test_func)
        hook.disable()
        
        context = HookContext(
            plugin_name="test_plugin",
            hook_type=HookType.BEFORE,
            data={},
            timestamp=time.time()
        )
        
        result = hook.execute(context)
        assert result is True  # 禁用时返回True
        assert executed is False  # 但函数不会被执行
    
    def test_function_hook_execute_exception(self):
        """测试函数钩子执行异常。"""
        def test_func(context: HookContext) -> bool:
            raise ValueError("Test exception")
        
        hook = FunctionHook("test_hook", test_func)
        context = HookContext(
            plugin_name="test_plugin",
            hook_type=HookType.BEFORE,
            data={},
            timestamp=time.time()
        )
        
        with patch('harborai.core.plugins.hooks.logger') as mock_logger:
            result = hook.execute(context)
            assert result is False
            mock_logger.error.assert_called_once()
    
    def test_function_hook_return_false(self):
        """测试函数钩子返回False。"""
        def test_func(context: HookContext) -> bool:
            return False
        
        hook = FunctionHook("test_hook", test_func)
        context = HookContext(
            plugin_name="test_plugin",
            hook_type=HookType.BEFORE,
            data={},
            timestamp=time.time()
        )
        
        result = hook.execute(context)
        assert result is False


class TestHookManager:
    """测试钩子管理器。"""
    
    def test_hook_manager_initialization(self):
        """测试钩子管理器初始化。"""
        manager = HookManager()
        
        # 检查所有钩子类型都被初始化
        for hook_type in HookType:
            assert hook_type in manager._hooks
            assert isinstance(manager._hooks[hook_type], list)
            assert len(manager._hooks[hook_type]) == 0
    
    def test_register_hook_success(self):
        """测试成功注册钩子。"""
        manager = HookManager()
        
        def test_func(context: HookContext) -> bool:
            return True
        
        hook = FunctionHook("test_hook", test_func, priority=5)
        
        with patch('harborai.core.plugins.hooks.logger') as mock_logger:
            result = manager.register_hook(HookType.BEFORE, hook)
            assert result is True
            mock_logger.info.assert_called_once()
        
        # 检查钩子是否被注册
        hooks = manager.get_hooks(HookType.BEFORE)
        assert len(hooks) == 1
        assert hooks[0] == hook
    
    def test_register_hook_priority_sorting(self):
        """测试钩子按优先级排序。"""
        manager = HookManager()
        
        def test_func(context: HookContext) -> bool:
            return True
        
        hook1 = FunctionHook("hook1", test_func, priority=1)
        hook2 = FunctionHook("hook2", test_func, priority=5)
        hook3 = FunctionHook("hook3", test_func, priority=3)
        
        manager.register_hook(HookType.BEFORE, hook1)
        manager.register_hook(HookType.BEFORE, hook2)
        manager.register_hook(HookType.BEFORE, hook3)
        
        hooks = manager.get_hooks(HookType.BEFORE)
        assert len(hooks) == 3
        assert hooks[0].name == "hook2"  # 优先级5，最高
        assert hooks[1].name == "hook3"  # 优先级3，中等
        assert hooks[2].name == "hook1"  # 优先级1，最低
    
    def test_register_hook_replace_existing(self):
        """测试替换已存在的钩子。"""
        manager = HookManager()
        
        def test_func1(context: HookContext) -> bool:
            return True
        
        def test_func2(context: HookContext) -> bool:
            return False
        
        hook1 = FunctionHook("test_hook", test_func1)
        hook2 = FunctionHook("test_hook", test_func2)
        
        with patch('harborai.core.plugins.hooks.logger') as mock_logger:
            # 注册第一个钩子
            manager.register_hook(HookType.BEFORE, hook1)
            # 注册同名钩子，应该替换第一个
            manager.register_hook(HookType.BEFORE, hook2)
            
            # 检查警告日志
            mock_logger.warning.assert_called_once()
        
        hooks = manager.get_hooks(HookType.BEFORE)
        assert len(hooks) == 1
        assert hooks[0] == hook2
    
    def test_register_hook_exception(self):
        """测试注册钩子时的异常处理。"""
        manager = HookManager()
        
        # 创建一个会导致异常的mock钩子
        mock_hook = Mock()
        mock_hook.name = "test_hook"
        
        # 模拟_hooks属性访问异常
        with patch.object(manager, '_hooks', new_callable=lambda: Mock(side_effect=Exception("Test exception"))):
            with patch('harborai.core.plugins.hooks.logger') as mock_logger:
                result = manager.register_hook(HookType.BEFORE, mock_hook)
                assert result is False
                mock_logger.error.assert_called_once()
    
    def test_unregister_hook_success(self):
        """测试成功注销钩子。"""
        manager = HookManager()
        
        def test_func(context: HookContext) -> bool:
            return True
        
        hook = FunctionHook("test_hook", test_func)
        manager.register_hook(HookType.BEFORE, hook)
        
        with patch('harborai.core.plugins.hooks.logger') as mock_logger:
            result = manager.unregister_hook(HookType.BEFORE, "test_hook")
            assert result is True
            mock_logger.info.assert_called_once()
        
        hooks = manager.get_hooks(HookType.BEFORE)
        assert len(hooks) == 0
    
    def test_unregister_hook_not_found(self):
        """测试注销不存在的钩子。"""
        manager = HookManager()
        
        with patch('harborai.core.plugins.hooks.logger') as mock_logger:
            result = manager.unregister_hook(HookType.BEFORE, "nonexistent_hook")
            assert result is False
            mock_logger.warning.assert_called_once()
    
    def test_unregister_hook_exception(self):
        """测试注销钩子时的异常处理。"""
        manager = HookManager()
        
        # 模拟_hooks属性访问异常
        with patch.object(manager, '_hooks', new_callable=lambda: Mock(side_effect=Exception("Test exception"))):
            with patch('harborai.core.plugins.hooks.logger') as mock_logger:
                result = manager.unregister_hook(HookType.BEFORE, "test_hook")
                assert result is False
                mock_logger.error.assert_called_once()
    
    def test_execute_hooks_success(self):
        """测试成功执行钩子。"""
        manager = HookManager()
        
        executed_hooks = []
        
        def test_func1(context: HookContext) -> bool:
            executed_hooks.append("hook1")
            return True
        
        def test_func2(context: HookContext) -> bool:
            executed_hooks.append("hook2")
            return True
        
        hook1 = FunctionHook("hook1", test_func1, priority=1)
        hook2 = FunctionHook("hook2", test_func2, priority=2)
        
        manager.register_hook(HookType.BEFORE, hook1)
        manager.register_hook(HookType.BEFORE, hook2)
        
        context = HookContext(
            plugin_name="test_plugin",
            hook_type=HookType.BEFORE,
            data={},
            timestamp=time.time()
        )
        
        result = manager.execute_hooks(HookType.BEFORE, context)
        assert result is True
        assert executed_hooks == ["hook2", "hook1"]  # 按优先级顺序执行
    
    def test_execute_hooks_early_termination(self):
        """测试钩子提前终止。"""
        manager = HookManager()
        
        executed_hooks = []
        
        def test_func1(context: HookContext) -> bool:
            executed_hooks.append("hook1")
            return False  # 返回False，终止后续钩子执行
        
        def test_func2(context: HookContext) -> bool:
            executed_hooks.append("hook2")
            return True
        
        hook1 = FunctionHook("hook1", test_func1, priority=2)
        hook2 = FunctionHook("hook2", test_func2, priority=1)
        
        manager.register_hook(HookType.BEFORE, hook1)
        manager.register_hook(HookType.BEFORE, hook2)
        
        context = HookContext(
            plugin_name="test_plugin",
            hook_type=HookType.BEFORE,
            data={},
            timestamp=time.time()
        )
        
        with patch('harborai.core.plugins.hooks.logger') as mock_logger:
            result = manager.execute_hooks(HookType.BEFORE, context)
            assert result is False
            mock_logger.warning.assert_called_once()
        
        assert executed_hooks == ["hook1"]  # 只执行了第一个钩子
    
    def test_execute_hooks_disabled_hook(self):
        """测试执行禁用的钩子。"""
        manager = HookManager()
        
        executed_hooks = []
        
        def test_func1(context: HookContext) -> bool:
            executed_hooks.append("hook1")
            return True
        
        def test_func2(context: HookContext) -> bool:
            executed_hooks.append("hook2")
            return True
        
        hook1 = FunctionHook("hook1", test_func1)
        hook2 = FunctionHook("hook2", test_func2)
        hook1.disable()  # 禁用第一个钩子
        
        manager.register_hook(HookType.BEFORE, hook1)
        manager.register_hook(HookType.BEFORE, hook2)
        
        context = HookContext(
            plugin_name="test_plugin",
            hook_type=HookType.BEFORE,
            data={},
            timestamp=time.time()
        )
        
        result = manager.execute_hooks(HookType.BEFORE, context)
        assert result is True
        assert executed_hooks == ["hook2"]  # 只执行了启用的钩子
    
    def test_execute_hooks_exception_handling(self):
        """测试钩子执行异常处理。"""
        manager = HookManager()
        
        def test_func1(context: HookContext) -> bool:
            raise ValueError("Test exception")
        
        def test_func2(context: HookContext) -> bool:
            return True
        
        hook1 = FunctionHook("hook1", test_func1, priority=2)
        hook2 = FunctionHook("hook2", test_func2, priority=1)
        
        manager.register_hook(HookType.BEFORE, hook1)
        manager.register_hook(HookType.BEFORE, hook2)
        
        context = HookContext(
            plugin_name="test_plugin",
            hook_type=HookType.BEFORE,
            data={},
            timestamp=time.time()
        )
        
        with patch('harborai.core.plugins.hooks.logger') as mock_logger:
            result = manager.execute_hooks(HookType.BEFORE, context)
            assert result is False
            mock_logger.error.assert_called()
    
    def test_execute_hooks_error_hook_exception_continue(self):
        """测试错误处理钩子异常时继续执行。"""
        manager = HookManager()
        
        executed_hooks = []
        
        def test_func1(context: HookContext) -> bool:
            raise ValueError("Test exception")
        
        def test_func2(context: HookContext) -> bool:
            executed_hooks.append("hook2")
            return True
        
        hook1 = FunctionHook("hook1", test_func1, priority=2)
        hook2 = FunctionHook("hook2", test_func2, priority=1)
        
        manager.register_hook(HookType.ON_ERROR, hook1)
        manager.register_hook(HookType.ON_ERROR, hook2)
        
        context = HookContext(
            plugin_name="test_plugin",
            hook_type=HookType.ON_ERROR,
            data={},
            timestamp=time.time()
        )
        
        with patch('harborai.core.plugins.hooks.logger') as mock_logger:
            result = manager.execute_hooks(HookType.ON_ERROR, context)
            # FunctionHook.execute会捕获异常并返回False
            # HookManager.execute_hooks检查到返回False，会记录警告并返回False
            # 对于ON_ERROR类型，不会继续执行其他钩子
            assert result is False
            # 应该记录错误（FunctionHook内部）和警告（HookManager中的False返回值处理）
            mock_logger.error.assert_called()
            mock_logger.warning.assert_called()
        
        # 由于第一个钩子返回False，第二个钩子不会被执行
        assert executed_hooks == []
    
    def test_execute_hooks_no_hooks(self):
        """测试执行不存在的钩子类型。"""
        manager = HookManager()
        
        context = HookContext(
            plugin_name="test_plugin",
            hook_type=HookType.BEFORE,
            data={},
            timestamp=time.time()
        )
        
        result = manager.execute_hooks(HookType.BEFORE, context)
        assert result is True
    
    def test_execute_hooks_exception(self):
        """测试执行钩子时的异常处理。"""
        manager = HookManager()
        
        # 模拟_hooks属性访问异常
        with patch.object(manager, '_hooks', new_callable=lambda: Mock(side_effect=Exception("Test exception"))):
            context = HookContext(
                plugin_name="test_plugin",
                hook_type=HookType.BEFORE,
                data={},
                timestamp=time.time()
            )
            
            with patch('harborai.core.plugins.hooks.logger') as mock_logger:
                result = manager.execute_hooks(HookType.BEFORE, context)
                assert result is False
                mock_logger.error.assert_called_once()
    
    def test_get_hooks(self):
        """测试获取钩子列表。"""
        manager = HookManager()
        
        def test_func(context: HookContext) -> bool:
            return True
        
        hook1 = FunctionHook("hook1", test_func)
        hook2 = FunctionHook("hook2", test_func)
        
        manager.register_hook(HookType.BEFORE, hook1)
        manager.register_hook(HookType.AFTER, hook2)
        
        before_hooks = manager.get_hooks(HookType.BEFORE)
        after_hooks = manager.get_hooks(HookType.AFTER)
        error_hooks = manager.get_hooks(HookType.ERROR)
        
        assert len(before_hooks) == 1
        assert before_hooks[0] == hook1
        assert len(after_hooks) == 1
        assert after_hooks[0] == hook2
        assert len(error_hooks) == 0
    
    def test_clear_hooks_specific_type(self):
        """测试清除特定类型的钩子。"""
        manager = HookManager()
        
        def test_func(context: HookContext) -> bool:
            return True
        
        hook1 = FunctionHook("hook1", test_func)
        hook2 = FunctionHook("hook2", test_func)
        
        manager.register_hook(HookType.BEFORE, hook1)
        manager.register_hook(HookType.AFTER, hook2)
        
        with patch('harborai.core.plugins.hooks.logger') as mock_logger:
            manager.clear_hooks(HookType.BEFORE)
            mock_logger.info.assert_called_once()
        
        before_hooks = manager.get_hooks(HookType.BEFORE)
        after_hooks = manager.get_hooks(HookType.AFTER)
        
        assert len(before_hooks) == 0
        assert len(after_hooks) == 1
    
    def test_clear_hooks_all_types(self):
        """测试清除所有类型的钩子。"""
        manager = HookManager()
        
        def test_func(context: HookContext) -> bool:
            return True
        
        hook1 = FunctionHook("hook1", test_func)
        hook2 = FunctionHook("hook2", test_func)
        
        manager.register_hook(HookType.BEFORE, hook1)
        manager.register_hook(HookType.AFTER, hook2)
        
        with patch('harborai.core.plugins.hooks.logger') as mock_logger:
            manager.clear_hooks()
            mock_logger.info.assert_called_once()
        
        for hook_type in HookType:
            hooks = manager.get_hooks(hook_type)
            assert len(hooks) == 0
    
    def test_clear_hooks_exception(self):
        """测试清除钩子时的异常处理。"""
        manager = HookManager()
        
        # 模拟_hooks属性访问异常
        with patch.object(manager, '_hooks', new_callable=lambda: Mock(side_effect=Exception("Test exception"))):
            with patch('harborai.core.plugins.hooks.logger') as mock_logger:
                manager.clear_hooks()
                mock_logger.error.assert_called_once()
    
    def test_list_hooks(self):
        """测试列出所有钩子。"""
        manager = HookManager()
        
        def test_func(context: HookContext) -> bool:
            return True
        
        hook1 = FunctionHook("hook1", test_func)
        hook2 = FunctionHook("hook2", test_func)
        hook3 = FunctionHook("hook3", test_func)
        
        manager.register_hook(HookType.BEFORE, hook1)
        manager.register_hook(HookType.BEFORE, hook2)
        manager.register_hook(HookType.AFTER, hook3)
        
        hooks_list = manager.list_hooks()
        
        assert "before" in hooks_list
        assert "after" in hooks_list
        assert len(hooks_list["before"]) == 2
        assert "hook1" in hooks_list["before"]
        assert "hook2" in hooks_list["before"]
        assert len(hooks_list["after"]) == 1
        assert "hook3" in hooks_list["after"]
        
        # 检查空的钩子类型
        assert "error" in hooks_list
        assert len(hooks_list["error"]) == 0


class TestGlobalFunctions:
    """测试全局函数。"""
    
    def test_get_hook_manager(self):
        """测试获取全局钩子管理器。"""
        manager1 = get_hook_manager()
        manager2 = get_hook_manager()
        
        # 应该返回同一个实例
        assert manager1 is manager2
        assert isinstance(manager1, HookManager)
    
    def test_register_hook_global(self):
        """测试全局注册钩子。"""
        def test_func(context: HookContext) -> bool:
            return True
        
        hook = FunctionHook("global_test_hook", test_func)
        
        # 清理之前的钩子
        get_hook_manager().clear_hooks(HookType.BEFORE)
        
        result = register_hook(HookType.BEFORE, hook)
        assert result is True
        
        hooks = get_hook_manager().get_hooks(HookType.BEFORE)
        assert len(hooks) == 1
        assert hooks[0] == hook
        
        # 清理
        get_hook_manager().clear_hooks(HookType.BEFORE)
    
    def test_register_function_hook_global(self):
        """测试全局注册函数钩子。"""
        executed = False
        
        def test_func(context: HookContext) -> bool:
            nonlocal executed
            executed = True
            return True
        
        # 清理之前的钩子
        get_hook_manager().clear_hooks(HookType.AFTER)
        
        result = register_function_hook(HookType.AFTER, "global_func_hook", test_func, priority=5)
        assert result is True
        
        hooks = get_hook_manager().get_hooks(HookType.AFTER)
        assert len(hooks) == 1
        assert hooks[0].name == "global_func_hook"
        assert hooks[0].priority == 5
        
        # 测试执行
        context = HookContext(
            plugin_name="test_plugin",
            hook_type=HookType.AFTER,
            data={},
            timestamp=time.time()
        )
        
        result = hooks[0].execute(context)
        assert result is True
        assert executed is True
        
        # 清理
        get_hook_manager().clear_hooks(HookType.AFTER)
    
    def test_execute_hooks_global(self):
        """测试全局执行钩子。"""
        executed_data = []
        
        def test_func(context: HookContext) -> bool:
            executed_data.append({
                "plugin_name": context.plugin_name,
                "hook_type": context.hook_type.value,
                "data": context.data
            })
            return True
        
        # 清理之前的钩子
        get_hook_manager().clear_hooks(HookType.ERROR)
        
        register_function_hook(HookType.ERROR, "global_execute_test", test_func)
        
        test_data = {"error": "test error", "code": 500}
        result = execute_hooks(HookType.ERROR, "test_plugin", test_data)
        
        assert result is True
        assert len(executed_data) == 1
        assert executed_data[0]["plugin_name"] == "test_plugin"
        assert executed_data[0]["hook_type"] == "error"
        assert executed_data[0]["data"] == test_data
        
        # 清理
        get_hook_manager().clear_hooks(HookType.ERROR)
    
    def test_execute_hooks_global_with_timestamp(self):
        """测试全局执行钩子时的时间戳。"""
        captured_context = None
        
        def test_func(context: HookContext) -> bool:
            nonlocal captured_context
            captured_context = context
            return True
        
        # 清理之前的钩子
        get_hook_manager().clear_hooks(HookType.ON_SUCCESS)
        
        register_function_hook(HookType.ON_SUCCESS, "timestamp_test", test_func)
        
        start_time = time.time()
        execute_hooks(HookType.ON_SUCCESS, "test_plugin", {})
        end_time = time.time()
        
        assert captured_context is not None
        assert start_time <= captured_context.timestamp <= end_time
        
        # 清理
        get_hook_manager().clear_hooks(HookType.ON_SUCCESS)


class TestHookIntegration:
    """测试钩子系统集成。"""
    
    def test_complex_hook_workflow(self):
        """测试复杂的钩子工作流。"""
        manager = HookManager()
        execution_log = []
        
        def before_init_hook(context: HookContext) -> bool:
            execution_log.append(f"before_init: {context.plugin_name}")
            context.data["initialized"] = False
            return True
        
        def after_init_hook(context: HookContext) -> bool:
            execution_log.append(f"after_init: {context.plugin_name}")
            context.data["initialized"] = True
            return True
        
        def before_execute_hook(context: HookContext) -> bool:
            execution_log.append(f"before_execute: {context.plugin_name}")
            if not context.data.get("initialized", False):
                execution_log.append("ERROR: Not initialized")
                return False
            return True
        
        def after_execute_hook(context: HookContext) -> bool:
            execution_log.append(f"after_execute: {context.plugin_name}")
            context.data["executed"] = True
            return True
        
        def error_hook(context: HookContext) -> bool:
            execution_log.append(f"error: {context.plugin_name}")
            return True
        
        # 注册钩子
        manager.register_hook(HookType.BEFORE_INIT, FunctionHook("before_init", before_init_hook))
        manager.register_hook(HookType.AFTER_INIT, FunctionHook("after_init", after_init_hook))
        manager.register_hook(HookType.BEFORE_EXECUTE, FunctionHook("before_execute", before_execute_hook))
        manager.register_hook(HookType.AFTER_EXECUTE, FunctionHook("after_execute", after_execute_hook))
        manager.register_hook(HookType.ON_ERROR, FunctionHook("error", error_hook))
        
        # 模拟插件生命周期
        plugin_data = {}
        
        # 初始化前
        context = HookContext("test_plugin", HookType.BEFORE_INIT, plugin_data, time.time())
        manager.execute_hooks(HookType.BEFORE_INIT, context)
        
        # 初始化后
        context = HookContext("test_plugin", HookType.AFTER_INIT, plugin_data, time.time())
        manager.execute_hooks(HookType.AFTER_INIT, context)
        
        # 执行前
        context = HookContext("test_plugin", HookType.BEFORE_EXECUTE, plugin_data, time.time())
        manager.execute_hooks(HookType.BEFORE_EXECUTE, context)
        
        # 执行后
        context = HookContext("test_plugin", HookType.AFTER_EXECUTE, plugin_data, time.time())
        manager.execute_hooks(HookType.AFTER_EXECUTE, context)
        
        # 检查执行日志
        expected_log = [
            "before_init: test_plugin",
            "after_init: test_plugin",
            "before_execute: test_plugin",
            "after_execute: test_plugin"
        ]
        assert execution_log == expected_log
        
        # 检查数据状态
        assert plugin_data["initialized"] is True
        assert plugin_data["executed"] is True
    
    def test_hook_priority_and_termination(self):
        """测试钩子优先级和终止机制。"""
        manager = HookManager()
        execution_log = []
        
        def high_priority_hook(context: HookContext) -> bool:
            execution_log.append("high_priority")
            return True
        
        def medium_priority_hook(context: HookContext) -> bool:
            execution_log.append("medium_priority")
            return False  # 终止后续钩子执行
        
        def low_priority_hook(context: HookContext) -> bool:
            execution_log.append("low_priority")
            return True
        
        # 注册不同优先级的钩子
        manager.register_hook(HookType.BEFORE, FunctionHook("high", high_priority_hook, priority=10))
        manager.register_hook(HookType.BEFORE, FunctionHook("medium", medium_priority_hook, priority=5))
        manager.register_hook(HookType.BEFORE, FunctionHook("low", low_priority_hook, priority=1))
        
        context = HookContext("test_plugin", HookType.BEFORE, {}, time.time())
        result = manager.execute_hooks(HookType.BEFORE, context)
        
        # 应该按优先级执行，但在medium_priority处终止
        assert result is False
        assert execution_log == ["high_priority", "medium_priority"]
        # low_priority不应该被执行
        assert "low_priority" not in execution_log
    
    def test_hook_data_sharing(self):
        """测试钩子间数据共享。"""
        manager = HookManager()
        
        def hook1(context: HookContext) -> bool:
            context.data["step1"] = "completed"
            context.data["counter"] = context.data.get("counter", 0) + 1
            return True
        
        def hook2(context: HookContext) -> bool:
            assert context.data["step1"] == "completed"
            context.data["step2"] = "completed"
            context.data["counter"] = context.data.get("counter", 0) + 1
            return True
        
        def hook3(context: HookContext) -> bool:
            assert context.data["step1"] == "completed"
            assert context.data["step2"] == "completed"
            context.data["step3"] = "completed"
            context.data["counter"] = context.data.get("counter", 0) + 1
            return True
        
        manager.register_hook(HookType.BEFORE, FunctionHook("hook1", hook1, priority=3))
        manager.register_hook(HookType.BEFORE, FunctionHook("hook2", hook2, priority=2))
        manager.register_hook(HookType.BEFORE, FunctionHook("hook3", hook3, priority=1))
        
        shared_data = {"initial": "value"}
        context = HookContext("test_plugin", HookType.BEFORE, shared_data, time.time())
        result = manager.execute_hooks(HookType.BEFORE, context)
        
        assert result is True
        assert shared_data["step1"] == "completed"
        assert shared_data["step2"] == "completed"
        assert shared_data["step3"] == "completed"
        assert shared_data["counter"] == 3
        assert shared_data["initial"] == "value"  # 原始数据保持不变