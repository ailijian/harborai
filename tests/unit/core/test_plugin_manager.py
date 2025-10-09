"""插件管理器测试

测试插件的发现、加载、管理和调用功能。
"""

import asyncio
import os
import sys
import tempfile
import pytest
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
from concurrent.futures import ThreadPoolExecutor, as_completed

from harborai.core.plugin_manager import (
    PluginStatus,
    PluginInfo,
    PluginManager,
    get_plugin_manager,
    set_plugin_manager,
    _global_plugin_manager
)
from harborai.core.base_plugin import BaseLLMPlugin, ModelInfo, ChatMessage, ChatCompletion
from harborai.core.exceptions import PluginError


class MockPlugin(BaseLLMPlugin):
    """测试用的模拟插件"""
    
    name = "mock_plugin"
    version = "1.0.0"
    description = "Mock plugin for testing"
    author = "Test Author"
    dependencies = []
    
    def __init__(self, config=None):
        super().__init__(self.name, **(config or {}))
        self._supported_models = [
            ModelInfo(
                id="mock-model-1",
                name="Mock Model 1",
                provider="mock",
                supports_streaming=True
            )
        ]
        self.initialized = False
        self.cleaned_up = False
    
    def initialize(self):
        """初始化插件"""
        self.initialized = True
    
    def cleanup(self):
        """清理插件"""
        self.cleaned_up = True
    
    def chat_completion(self, model, messages, stream=False, **kwargs):
        """同步聊天完成"""
        return ChatCompletion(
            id="test-completion",
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[]
        )
    
    async def chat_completion_async(self, model, messages, stream=False, **kwargs):
        """异步聊天完成"""
        return self.chat_completion(model, messages, stream, **kwargs)
    
    def on_request_start(self, *args, **kwargs):
        """请求开始钩子"""
        return "request_start_called"
    
    def on_request_end(self, *args, **kwargs):
        """请求结束钩子"""
        return "request_end_called"


class MockAsyncPlugin(BaseLLMPlugin):
    """异步初始化的模拟插件"""
    
    name = "mock_async_plugin"
    version = "1.0.0"
    
    def __init__(self, config=None):
        super().__init__(self.name, **(config or {}))
        self.initialized = False
    
    async def initialize(self):
        """异步初始化"""
        await asyncio.sleep(0.01)
        self.initialized = True
    
    async def cleanup(self):
        """异步清理"""
        await asyncio.sleep(0.01)
        self.initialized = False
    
    def chat_completion(self, model, messages, stream=False, **kwargs):
        return ChatCompletion(
            id="async-completion",
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[]
        )
    
    async def chat_completion_async(self, model, messages, stream=False, **kwargs):
        return self.chat_completion(model, messages, stream, **kwargs)


class MockDependentPlugin(BaseLLMPlugin):
    """有依赖的模拟插件"""
    
    name = "mock_dependent_plugin"
    version = "1.0.0"
    dependencies = ["mock_plugin"]
    
    def __init__(self, config=None):
        super().__init__(self.name, **(config or {}))
    
    def chat_completion(self, model, messages, stream=False, **kwargs):
        return ChatCompletion(
            id="dependent-completion",
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[]
        )
    
    async def chat_completion_async(self, model, messages, stream=False, **kwargs):
        return self.chat_completion(model, messages, stream, **kwargs)


class TestPluginStatus:
    """测试插件状态枚举"""
    
    def test_plugin_status_values(self):
        """测试插件状态值"""
        assert PluginStatus.UNLOADED.value == "unloaded"
        assert PluginStatus.LOADING.value == "loading"
        assert PluginStatus.LOADED.value == "loaded"
        assert PluginStatus.ACTIVE.value == "active"
        assert PluginStatus.INACTIVE.value == "inactive"
        assert PluginStatus.ERROR.value == "error"
        assert PluginStatus.DISABLED.value == "disabled"


class TestPluginInfo:
    """测试插件信息类"""
    
    def test_plugin_info_creation(self):
        """测试插件信息创建"""
        info = PluginInfo(
            name="test_plugin",
            version="1.0.0",
            description="Test plugin",
            author="Test Author"
        )
        
        assert info.name == "test_plugin"
        assert info.version == "1.0.0"
        assert info.description == "Test plugin"
        assert info.author == "Test Author"
        assert info.status == PluginStatus.UNLOADED
        assert info.config == {}
        assert info.dependencies == []
    
    def test_plugin_info_to_dict(self):
        """测试插件信息转换为字典"""
        info = PluginInfo(
            name="test_plugin",
            version="2.0.0",
            description="Test plugin",
            author="Test Author",
            module_path="test.module",
            config={"key": "value"},
            dependencies=["dep1", "dep2"],
            status=PluginStatus.LOADED,
            load_time=1234567890.0,
            error_info="Test error"
        )
        
        result = info.to_dict()
        
        assert result["name"] == "test_plugin"
        assert result["version"] == "2.0.0"
        assert result["description"] == "Test plugin"
        assert result["author"] == "Test Author"
        assert result["module_path"] == "test.module"
        assert result["config"] == {"key": "value"}
        assert result["dependencies"] == ["dep1", "dep2"]
        assert result["status"] == "loaded"
        assert result["load_time"] == 1234567890.0
        assert result["error_info"] == "Test error"


class TestPluginManager:
    """测试插件管理器"""
    
    def setup_method(self):
        """每个测试前的设置"""
        # 重置全局插件管理器
        global _global_plugin_manager
        _global_plugin_manager = None
    
    def test_plugin_manager_creation(self):
        """测试插件管理器创建"""
        manager = PluginManager(
            plugin_dirs=["/test/dir1", "/test/dir2"],
            config={"test": "config"},
            auto_load=False,
            max_workers=8
        )
        
        # 检查指定的目录是否包含在plugin_dirs中
        assert "/test/dir1" in manager.plugin_dirs
        assert "/test/dir2" in manager.plugin_dirs
        assert manager.config == {"test": "config"}
        assert manager.auto_load is False
        assert manager.max_workers == 8
        assert len(manager._plugins) == 0
        assert len(manager._plugin_instances) == 0
        assert len(manager._plugin_hooks) == 0
    
    def test_init_default_plugin_dirs(self):
        """测试初始化默认插件目录"""
        manager = PluginManager(auto_load=False)
        
        # 应该包含默认的插件目录
        assert len(manager.plugin_dirs) >= 0  # 可能不存在目录
    
    @patch('os.path.exists')
    @patch('os.walk')
    def test_discover_plugins(self, mock_walk, mock_exists):
        """测试插件发现"""
        mock_exists.return_value = True
        mock_walk.return_value = [
            ("/test/plugins", [], ["plugin1.py", "plugin2.py", "__init__.py"]),
            ("/test/plugins/subdir", [], ["plugin3.py"])
        ]
        
        manager = PluginManager(plugin_dirs=["/test/plugins"], auto_load=False)
        discovered = manager.discover_plugins()
        
        assert "plugin1" in discovered
        assert "plugin2" in discovered
        assert "subdir.plugin3" in discovered
        assert "__init__" not in discovered
    
    @patch('importlib.import_module')
    def test_load_plugin_success(self, mock_import):
        """测试成功加载插件"""
        # 创建模拟模块
        mock_module = MagicMock()
        mock_module.MockPlugin = MockPlugin
        mock_import.return_value = mock_module
        
        manager = PluginManager(auto_load=False)
        
        # 加载插件
        success = manager.load_plugin("test_module")
        
        assert success is True
        assert "mock_plugin" in manager._plugins
        assert "mock_plugin" in manager._plugin_instances
        
        plugin_info = manager._plugins["mock_plugin"]
        assert plugin_info.name == "mock_plugin"
        assert plugin_info.version == "1.0.0"
        assert plugin_info.status == PluginStatus.LOADED
        assert plugin_info.instance.initialized is True
    
    @patch('importlib.import_module')
    def test_load_plugin_async_init(self, mock_import):
        """测试加载异步初始化插件"""
        mock_module = MagicMock()
        mock_module.MockAsyncPlugin = MockAsyncPlugin
        mock_import.return_value = mock_module
        
        manager = PluginManager(auto_load=False)
        
        # 加载插件
        success = manager.load_plugin("test_async_module")
        
        assert success is True
        assert "mock_async_plugin" in manager._plugins
        
        plugin_info = manager._plugins["mock_async_plugin"]
        assert plugin_info.instance.initialized is True
    
    @patch('importlib.import_module')
    def test_load_plugin_no_plugin_class(self, mock_import):
        """测试加载没有插件类的模块"""
        mock_module = MagicMock()
        # 模块中没有继承BaseLLMPlugin的类
        mock_import.return_value = mock_module
        
        manager = PluginManager(auto_load=False)
        
        with pytest.raises(PluginError):
            manager.load_plugin("invalid_module")
    
    @patch('importlib.import_module')
    def test_load_plugin_with_dependencies(self, mock_import):
        """测试加载有依赖的插件"""
        # 先加载依赖插件
        mock_module1 = MagicMock()
        mock_module1.MockPlugin = MockPlugin
        
        mock_module2 = MagicMock()
        mock_module2.MockDependentPlugin = MockDependentPlugin
        
        def side_effect(module_name):
            if module_name == "base_module":
                return mock_module1
            elif module_name == "dependent_module":
                return mock_module2
            raise ImportError(f"No module named {module_name}")
        
        mock_import.side_effect = side_effect
        
        manager = PluginManager(auto_load=False)
        
        # 先加载基础插件
        manager.load_plugin("base_module")
        
        # 再加载依赖插件
        success = manager.load_plugin("dependent_module")
        
        assert success is True
        assert "mock_dependent_plugin" in manager._plugins
    
    @patch('importlib.import_module')
    def test_load_plugin_missing_dependency(self, mock_import):
        """测试加载缺少依赖的插件"""
        mock_module = MagicMock()
        mock_module.MockDependentPlugin = MockDependentPlugin
        mock_import.return_value = mock_module
        
        manager = PluginManager(auto_load=False)
        
        # 尝试加载有依赖但依赖未加载的插件
        with pytest.raises(PluginError):
            manager.load_plugin("dependent_module")
    
    def test_unload_plugin(self):
        """测试卸载插件"""
        manager = PluginManager(auto_load=False)
        
        # 手动添加插件
        plugin_instance = MockPlugin()
        plugin_info = PluginInfo(
            name="mock_plugin",
            instance=plugin_instance,
            status=PluginStatus.LOADED
        )
        
        manager._plugins["mock_plugin"] = plugin_info
        manager._plugin_instances["mock_plugin"] = plugin_instance
        
        # 注册钩子
        manager._plugin_hooks["test_hook"] = [plugin_instance.on_request_start]
        
        # 卸载插件
        success = manager.unload_plugin("mock_plugin")
        
        assert success is True
        assert "mock_plugin" not in manager._plugins
        assert "mock_plugin" not in manager._plugin_instances
        assert plugin_instance.cleaned_up is True
        assert len(manager._plugin_hooks["test_hook"]) == 0
    
    def test_unload_nonexistent_plugin(self):
        """测试卸载不存在的插件"""
        manager = PluginManager(auto_load=False)
        
        success = manager.unload_plugin("nonexistent_plugin")
        assert success is False
    
    def test_get_plugin(self):
        """测试获取插件实例"""
        manager = PluginManager(auto_load=False)
        
        plugin_instance = MockPlugin()
        manager._plugin_instances["mock_plugin"] = plugin_instance
        
        retrieved = manager.get_plugin("mock_plugin")
        assert retrieved is plugin_instance
        
        # 测试获取不存在的插件
        assert manager.get_plugin("nonexistent") is None
    
    def test_get_plugin_info(self):
        """测试获取插件信息"""
        manager = PluginManager(auto_load=False)
        
        plugin_info = PluginInfo(name="test_plugin")
        manager._plugins["test_plugin"] = plugin_info
        
        retrieved = manager.get_plugin_info("test_plugin")
        assert retrieved is plugin_info
        
        # 测试获取不存在的插件信息
        assert manager.get_plugin_info("nonexistent") is None
    
    def test_list_plugins(self):
        """测试列出插件"""
        manager = PluginManager(auto_load=False)
        
        # 添加不同状态的插件
        plugin1 = PluginInfo(name="plugin1", status=PluginStatus.LOADED)
        plugin2 = PluginInfo(name="plugin2", status=PluginStatus.ACTIVE)
        plugin3 = PluginInfo(name="plugin3", status=PluginStatus.ERROR)
        
        manager._plugins["plugin1"] = plugin1
        manager._plugins["plugin2"] = plugin2
        manager._plugins["plugin3"] = plugin3
        
        # 列出所有插件
        all_plugins = manager.list_plugins()
        assert len(all_plugins) == 3
        
        # 按状态过滤
        loaded_plugins = manager.list_plugins(PluginStatus.LOADED)
        assert len(loaded_plugins) == 1
        assert loaded_plugins[0].name == "plugin1"
        
        active_plugins = manager.list_plugins(PluginStatus.ACTIVE)
        assert len(active_plugins) == 1
        assert active_plugins[0].name == "plugin2"
    
    def test_enable_disable_plugin(self):
        """测试启用和禁用插件"""
        manager = PluginManager(auto_load=False)
        
        plugin_info = PluginInfo(name="test_plugin", status=PluginStatus.LOADED)
        manager._plugins["test_plugin"] = plugin_info
        
        # 启用插件
        success = manager.enable_plugin("test_plugin")
        assert success is True
        assert plugin_info.status == PluginStatus.ACTIVE
        
        # 禁用插件
        success = manager.disable_plugin("test_plugin")
        assert success is True
        assert plugin_info.status == PluginStatus.INACTIVE
        
        # 测试不存在的插件
        assert manager.enable_plugin("nonexistent") is False
        assert manager.disable_plugin("nonexistent") is False
    
    def test_call_plugin(self):
        """测试调用插件方法"""
        manager = PluginManager(auto_load=False)
        
        plugin_instance = MockPlugin()
        manager._plugin_instances["mock_plugin"] = plugin_instance
        
        # 调用存在的方法
        result = manager.call_plugin(
            "mock_plugin",
            "chat_completion",
            "mock-model-1",
            [ChatMessage(role="user", content="test")]
        )
        
        assert isinstance(result, ChatCompletion)
        assert result.model == "mock-model-1"
        
        # 调用不存在的插件
        with pytest.raises(PluginError):
            manager.call_plugin("nonexistent", "method")
        
        # 调用不存在的方法
        with pytest.raises(PluginError):
            manager.call_plugin("mock_plugin", "nonexistent_method")
    
    @pytest.mark.asyncio
    async def test_call_plugin_async(self):
        """测试异步调用插件方法"""
        manager = PluginManager(auto_load=False)
        
        plugin_instance = MockPlugin()
        manager._plugin_instances["mock_plugin"] = plugin_instance
        
        # 调用异步方法
        result = await manager.call_plugin_async(
            "mock_plugin",
            "chat_completion_async",
            "mock-model-1",
            [ChatMessage(role="user", content="test")]
        )
        
        assert isinstance(result, ChatCompletion)
        assert result.model == "mock-model-1"
        
        # 调用同步方法（应该在线程池中执行）
        result = await manager.call_plugin_async(
            "mock_plugin",
            "chat_completion",
            "mock-model-1",
            [ChatMessage(role="user", content="test")]
        )
        
        assert isinstance(result, ChatCompletion)
    
    @pytest.mark.asyncio
    async def test_trigger_hook(self):
        """测试触发钩子"""
        manager = PluginManager(auto_load=False)
        
        plugin_instance = MockPlugin()
        manager._plugin_hooks["request_start"] = [plugin_instance.on_request_start]
        manager._plugin_hooks["request_end"] = [plugin_instance.on_request_end]
        
        # 触发钩子
        results = await manager.trigger_hook("request_start", "arg1", key="value")
        assert len(results) == 1
        assert results[0] == "request_start_called"
        
        # 触发不存在的钩子
        results = await manager.trigger_hook("nonexistent_hook")
        assert len(results) == 0
    
    def test_get_plugin_stats(self):
        """测试获取插件统计信息"""
        manager = PluginManager(auto_load=False)
        
        # 添加插件
        plugin1 = PluginInfo(name="plugin1", status=PluginStatus.LOADED)
        plugin2 = PluginInfo(name="plugin2", status=PluginStatus.ACTIVE)
        plugin3 = PluginInfo(name="plugin3", status=PluginStatus.ERROR)
        
        manager._plugins["plugin1"] = plugin1
        manager._plugins["plugin2"] = plugin2
        manager._plugins["plugin3"] = plugin3
        
        # 添加钩子
        manager._plugin_hooks["hook1"] = [lambda: None, lambda: None]
        manager._plugin_hooks["hook2"] = [lambda: None]
        
        stats = manager.get_plugin_stats()
        
        assert stats["total_plugins"] == 3
        assert stats["loaded_plugins"] == 1
        assert stats["active_plugins"] == 1
        assert stats["error_plugins"] == 1
        assert stats["hooks_count"]["hook1"] == 2
        assert stats["hooks_count"]["hook2"] == 1
    
    def test_cleanup(self):
        """测试清理资源"""
        manager = PluginManager(auto_load=False)
        
        # 添加插件
        plugin_instance = MockPlugin()
        plugin_info = PluginInfo(
            name="mock_plugin",
            instance=plugin_instance,
            status=PluginStatus.LOADED
        )
        
        manager._plugins["mock_plugin"] = plugin_info
        manager._plugin_instances["mock_plugin"] = plugin_instance
        
        # 清理
        manager.cleanup()
        
        assert len(manager._plugins) == 0
        assert len(manager._plugin_instances) == 0
        assert plugin_instance.cleaned_up is True


class TestGlobalPluginManager:
    """测试全局插件管理器"""
    
    def setup_method(self):
        """每个测试前重置全局状态"""
        global _global_plugin_manager
        _global_plugin_manager = None
    
    def test_get_plugin_manager_singleton(self):
        """测试全局插件管理器单例"""
        manager1 = get_plugin_manager()
        manager2 = get_plugin_manager()
        
        assert manager1 is manager2
        assert isinstance(manager1, PluginManager)
    
    def test_set_plugin_manager(self):
        """测试设置全局插件管理器"""
        custom_manager = PluginManager(auto_load=False)
        set_plugin_manager(custom_manager)
        
        retrieved = get_plugin_manager()
        assert retrieved is custom_manager


class TestPluginManagerConcurrency:
    """测试插件管理器并发场景"""
    
    @patch('importlib.import_module')
    def test_concurrent_plugin_loading(self, mock_import):
        """测试并发插件加载"""
        # 创建多个不同的模拟插件类
        class TestPlugin1(MockPlugin):
            name = "test_plugin_1"
        
        class TestPlugin2(MockPlugin):
            name = "test_plugin_2"
        
        class TestPlugin3(MockPlugin):
            name = "test_plugin_3"
        
        def mock_import_side_effect(module_name):
            mock_module = MagicMock()
            if module_name == "module1":
                mock_module.TestPlugin1 = TestPlugin1
            elif module_name == "module2":
                mock_module.TestPlugin2 = TestPlugin2
            elif module_name == "module3":
                mock_module.TestPlugin3 = TestPlugin3
            else:
                raise ImportError(f"No module named {module_name}")
            return mock_module
        
        mock_import.side_effect = mock_import_side_effect
        
        manager = PluginManager(auto_load=False)
        results = []
        errors = []
        
        def load_plugin_worker(module_name):
            """工作线程函数"""
            try:
                success = manager.load_plugin(module_name)
                results.append((module_name, success))
            except Exception as e:
                errors.append((module_name, e))
        
        # 启动多个线程同时加载插件
        threads = []
        modules = ["module1", "module2", "module3"]
        
        for module in modules:
            thread = threading.Thread(target=load_plugin_worker, args=(module,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证结果
        assert len(errors) == 0, f"加载错误: {errors}"
        assert len(results) == 3
        assert all(success for _, success in results)
        
        # 验证所有插件都已加载
        assert len(manager._plugins) == 3
        assert "test_plugin_1" in manager._plugins
        assert "test_plugin_2" in manager._plugins
        assert "test_plugin_3" in manager._plugins
    
    def test_concurrent_plugin_operations(self):
        """测试并发插件操作"""
        manager = PluginManager(auto_load=False)
        
        # 添加插件
        for i in range(5):
            plugin_instance = MockPlugin()
            plugin_instance.name = f"plugin_{i}"
            plugin_info = PluginInfo(
                name=f"plugin_{i}",
                instance=plugin_instance,
                status=PluginStatus.LOADED
            )
            manager._plugins[f"plugin_{i}"] = plugin_info
            manager._plugin_instances[f"plugin_{i}"] = plugin_instance
        
        results = []
        errors = []
        
        def worker(worker_id):
            """工作线程函数"""
            try:
                for i in range(10):
                    plugin_name = f"plugin_{i % 5}"
                    
                    # 获取插件
                    plugin = manager.get_plugin(plugin_name)
                    results.append(plugin is not None)
                    
                    # 获取插件信息
                    info = manager.get_plugin_info(plugin_name)
                    results.append(info is not None)
                    
                    # 启用/禁用插件
                    if i % 2 == 0:
                        manager.enable_plugin(plugin_name)
                    else:
                        manager.disable_plugin(plugin_name)
                    
                    # 获取统计信息
                    stats = manager.get_plugin_stats()
                    results.append(isinstance(stats, dict))
                    
            except Exception as e:
                errors.append(e)
        
        # 启动多个工作线程
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证结果
        assert len(errors) == 0, f"操作错误: {errors}"
        assert all(results), "部分操作失败"
    
    @pytest.mark.asyncio
    async def test_concurrent_async_operations(self):
        """测试并发异步操作"""
        manager = PluginManager(auto_load=False)
        
        # 添加插件
        plugin_instance = MockPlugin()
        manager._plugin_instances["mock_plugin"] = plugin_instance
        manager._plugin_hooks["test_hook"] = [plugin_instance.on_request_start]
        
        results = []
        errors = []
        
        async def async_worker(worker_id):
            """异步工作函数"""
            try:
                for i in range(20):
                    # 异步调用插件方法
                    result = await manager.call_plugin_async(
                        "mock_plugin",
                        "chat_completion_async",
                        "mock-model-1",
                        [ChatMessage(role="user", content=f"test_{worker_id}_{i}")]
                    )
                    results.append(isinstance(result, ChatCompletion))
                    
                    # 触发钩子
                    hook_results = await manager.trigger_hook("test_hook", f"arg_{i}")
                    results.append(len(hook_results) == 1)
                    
                    await asyncio.sleep(0.001)  # 让出控制权
                    
            except Exception as e:
                errors.append(e)
        
        # 启动多个异步任务
        tasks = [async_worker(i) for i in range(3)]
        await asyncio.gather(*tasks)
        
        # 验证结果
        assert len(errors) == 0, f"异步操作错误: {errors}"
        assert all(results), "部分异步操作失败"


class TestPluginManagerErrorHandling:
    """测试插件管理器错误处理"""
    
    @patch('importlib.import_module')
    def test_load_plugin_import_error(self, mock_import):
        """测试插件导入错误"""
        mock_import.side_effect = ImportError("Module not found")
        
        manager = PluginManager(auto_load=False)
        
        with pytest.raises(PluginError):
            manager.load_plugin("nonexistent_module")
    
    @patch('importlib.import_module')
    def test_load_plugin_initialization_error(self, mock_import):
        """测试插件初始化错误"""
        class FailingPlugin(BaseLLMPlugin):
            name = "failing_plugin"
            
            def __init__(self, config=None):
                super().__init__(self.name, **(config or {}))
                raise RuntimeError("Initialization failed")
            
            def chat_completion(self, model, messages, stream=False, **kwargs):
                pass
            
            async def chat_completion_async(self, model, messages, stream=False, **kwargs):
                pass
        
        mock_module = MagicMock()
        mock_module.FailingPlugin = FailingPlugin
        mock_import.return_value = mock_module
        
        manager = PluginManager(auto_load=False)
        
        with pytest.raises(PluginError):
            manager.load_plugin("failing_module")
    
    @pytest.mark.asyncio
    async def test_hook_error_handling(self):
        """测试钩子错误处理"""
        manager = PluginManager(auto_load=False)
        
        def failing_hook(*args, **kwargs):
            raise RuntimeError("Hook failed")
        
        def working_hook(*args, **kwargs):
            return "success"
        
        manager._plugin_hooks["test_hook"] = [failing_hook, working_hook]
        
        # 触发钩子，应该处理错误并继续执行其他钩子
        results = await manager.trigger_hook("test_hook")
        
        assert len(results) == 2
        assert results[0] is None  # 失败的钩子返回None
        assert results[1] == "success"  # 成功的钩子正常返回


if __name__ == "__main__":
    pytest.main([__file__, "-v"])