import pytest
from unittest.mock import Mock, patch, MagicMock
from harborai.core.client_manager import ClientManager
from harborai.core.exceptions import PluginError, ModelNotFoundError
from harborai.config.settings import Settings

class TestClientManagerAdditionalCoverage:
    """测试ClientManager的额外覆盖率"""
    
    @pytest.fixture
    def settings(self):
        """创建测试用的设置"""
        settings = Mock(spec=Settings)
        settings.plugin_directories = ["harborai.plugins"]
        settings.model_mappings = {"test-model": "actual-model"}
        settings.get_plugin_config.return_value = {"api_key": "test-key"}
        return settings
    
    def test_scan_plugin_directory_import_error(self, settings):
        """测试扫描插件目录时的ImportError处理"""
        with patch('harborai.core.client_manager.get_settings', return_value=settings):
            manager = ClientManager(lazy_loading=False)
            
            with patch('importlib.import_module', side_effect=ImportError("Module not found")):
                # 这应该不会抛出异常，而是记录警告
                manager._scan_plugin_directory("non_existent_plugin")
    
    def test_load_plugin_module_instantiation_error(self, settings):
        """测试插件实例化时的异常处理"""
        with patch('harborai.core.client_manager.get_settings', return_value=settings):
            manager = ClientManager(lazy_loading=False)
            
            # 创建一个模拟的插件类
            from harborai.core.base_plugin import BaseLLMPlugin
            
            class FailingPlugin(BaseLLMPlugin):
                def __init__(self, *args, **kwargs):
                    raise Exception("Instantiation failed")
            
            # 模拟模块
            mock_module = Mock()
            mock_module.FailingPlugin = FailingPlugin
            
            with patch('importlib.import_module', return_value=mock_module):
                with patch('builtins.dir', return_value=['FailingPlugin']):
                    with patch('builtins.getattr', return_value=FailingPlugin):
                        # 这应该不会抛出异常，而是记录错误
                        manager._load_plugin_module("test_plugin")
    
    def test_register_plugin_duplicate_model(self, settings):
        """测试注册重复模型的插件"""
        with patch('harborai.core.client_manager.get_settings', return_value=settings):
            manager = ClientManager(lazy_loading=False)
            
            # 创建两个模拟插件，都支持相同的模型
            plugin1 = Mock()
            plugin1.name = "plugin1"
            plugin1.supported_models = [Mock(id="test-model")]
            
            plugin2 = Mock()
            plugin2.name = "plugin2"
            plugin2.supported_models = [Mock(id="test-model")]
            
            # 注册第一个插件
            manager.register_plugin(plugin1)
            
            # 注册第二个插件（应该记录警告）
            manager.register_plugin(plugin2)
            
            # 验证第二个插件覆盖了第一个
            assert manager.model_to_plugin["test-model"] == "plugin2"
    
    def test_unregister_plugin_not_found(self, settings):
        """测试注销不存在的插件"""
        with patch('harborai.core.client_manager.get_settings', return_value=settings):
            manager = ClientManager(lazy_loading=False)
            
            # 验证抛出正确的异常类型，但不检查异常消息
            with pytest.raises(PluginError):
                manager.unregister_plugin("non_existent_plugin")
    
    def test_get_plugin_for_model_with_mapping(self, settings):
        """测试通过模型映射获取插件"""
        with patch('harborai.core.client_manager.get_settings', return_value=settings):
            manager = ClientManager(lazy_loading=False)
            
            # 创建一个插件
            plugin = Mock()
            plugin.name = "test_plugin"
            plugin.supported_models = [Mock(id="actual-model")]
            
            manager.register_plugin(plugin)
            
            # 通过映射获取插件
            result = manager.get_plugin_for_model("test-model")
            assert result == plugin
    
    def test_get_plugin_for_model_not_found(self, settings):
        """测试获取不存在的模型插件"""
        with patch('harborai.core.client_manager.get_settings', return_value=settings):
            manager = ClientManager(lazy_loading=False)
            
            # 验证抛出正确的异常类型，但不检查异常消息
            with pytest.raises(ModelNotFoundError):
                manager.get_plugin_for_model("non_existent_model")
    
    def test_lazy_loading_mode_initialization(self, settings):
        """测试延迟加载模式的初始化"""
        with patch('harborai.core.client_manager.get_settings', return_value=settings):
            with patch('harborai.core.client_manager.LazyPluginManager') as mock_lazy_manager:
                manager = ClientManager(lazy_loading=True)
                
                # 验证LazyPluginManager被创建
                mock_lazy_manager.assert_called_once()
                assert manager._lazy_manager is not None
    
    def test_get_available_models_lazy_mode(self, settings):
        """测试延迟模式下获取可用模型"""
        with patch('harborai.core.client_manager.get_settings', return_value=settings):
            with patch('harborai.core.client_manager.LazyPluginManager') as mock_lazy_manager_class:
                mock_lazy_manager = Mock()
                mock_lazy_manager.get_supported_models.return_value = ["model1", "model2"]
                mock_lazy_manager_class.return_value = mock_lazy_manager
                
                manager = ClientManager(lazy_loading=True)
                models = manager.get_available_models()
                
                # 验证返回的模型数量
                assert len(models) == 2
                # 验证模型ID
                assert models[0].id == "model1"
                assert models[1].id == "model2"
    
    def test_get_loading_statistics_lazy_mode(self, settings):
        """测试延迟模式下获取加载统计"""
        with patch('harborai.core.client_manager.get_settings', return_value=settings):
            with patch('harborai.core.client_manager.LazyPluginManager') as mock_lazy_manager_class:
                mock_lazy_manager = Mock()
                mock_lazy_manager.get_statistics.return_value = {"loaded": 2, "total": 5}
                mock_lazy_manager_class.return_value = mock_lazy_manager
                
                manager = ClientManager(lazy_loading=True)
                stats = manager.get_loading_statistics()
                
                assert stats == {"loaded": 2, "total": 5}
                mock_lazy_manager.get_statistics.assert_called_once()
    
    def test_get_plugin_info_lazy_mode(self, settings):
        """测试延迟模式下获取插件信息"""
        with patch('harborai.core.client_manager.get_settings', return_value=settings):
            with patch('harborai.core.client_manager.LazyPluginManager') as mock_lazy_manager_class:
                mock_lazy_manager = Mock()
                mock_lazy_manager.get_plugin_info.return_value = {"plugin1": {"loaded": True}}
                mock_lazy_manager_class.return_value = mock_lazy_manager
                
                manager = ClientManager(lazy_loading=True)
                info = manager.get_plugin_info()
                
                assert info == {"plugin1": {"loaded": True}}
                mock_lazy_manager.get_plugin_info.assert_called_once()
    
    def test_preload_plugin_non_lazy_mode(self, settings):
        """测试非延迟模式下的插件预加载"""
        with patch('harborai.core.client_manager.get_settings', return_value=settings):
            manager = ClientManager(lazy_loading=False)
            
            # 在非延迟模式下预加载应该产生警告，但不抛出异常
            manager.preload_plugin("test_plugin")
    
    def test_preload_model_non_lazy_mode(self, settings):
        """测试非延迟模式下的模型预加载"""
        with patch('harborai.core.client_manager.get_settings', return_value=settings):
            manager = ClientManager(lazy_loading=False)
            
            # 在非延迟模式下预加载应该产生警告，但不抛出异常
            manager.preload_model("test-model")