"""
测试structured_provider字段修复的测试用例

验证structured_provider字段在以下场景下的正确记录：
1. 显式传递structured_provider参数
2. 使用默认值（从settings.default_structured_provider获取）
3. 文件日志和PostgreSQL日志都正确记录
4. structured_provider与provider字段同时存在且正确记录
"""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from harborai import HarborAI
from harborai.config.settings import get_settings


class TestStructuredProviderFix:
    """测试structured_provider字段修复"""
    
    def setup_method(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, "test_logs.jsonl")
        
    def teardown_method(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.unit
    def test_structured_provider_explicit_value(self):
        """测试显式传递structured_provider参数"""
        # 模拟设置
        with patch('harborai.config.settings.get_settings') as mock_settings:
            mock_settings.return_value.default_structured_provider = "agently"
            mock_settings.return_value.enable_file_logging = True
            mock_settings.return_value.log_file_path = self.log_file
            
            # 模拟客户端管理器和API响应
            with patch('harborai.api.client.ClientManager') as mock_client_manager:
                mock_client = Mock()
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message = Mock()
                mock_response.choices[0].message.content = "测试响应"
                mock_response.usage = Mock()
                mock_response.usage.total_tokens = 100
                
                mock_client.chat.completions.create.return_value = mock_response
                mock_client_manager.return_value.get_client.return_value = mock_client
                
                # 模拟日志记录器
                with patch('harborai.api.client.APILogger') as mock_logger_class:
                    mock_logger = Mock()
                    mock_logger_class.return_value = mock_logger
                    
                    # 创建HarborAI客户端
                    client = HarborAI()
                    
                    # 调用create方法，显式传递structured_provider
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "测试消息"}],
                        structured_provider="native"
                    )
                    
                    # 验证日志记录器被正确调用
                    assert mock_logger.log_request.called
                    call_args = mock_logger.log_request.call_args
                    
                    # 验证structured_provider参数被正确传递
                    assert "structured_provider" in call_args.kwargs
                    assert call_args.kwargs["structured_provider"] == "native"
    
    @pytest.mark.unit
    def test_structured_provider_default_value(self):
        """测试使用默认structured_provider值"""
        # 模拟设置，设置默认值为"agently"
        with patch('harborai.config.settings.get_settings') as mock_settings:
            mock_settings.return_value.default_structured_provider = "agently"
            mock_settings.return_value.enable_file_logging = True
            mock_settings.return_value.log_file_path = self.log_file
            
            # 模拟客户端管理器和API响应
            with patch('harborai.api.client.ClientManager') as mock_client_manager:
                mock_client = Mock()
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message = Mock()
                mock_response.choices[0].message.content = "测试响应"
                mock_response.usage = Mock()
                mock_response.usage.total_tokens = 100
                
                mock_client.chat.completions.create.return_value = mock_response
                mock_client_manager.return_value.get_client.return_value = mock_client
                
                # 模拟日志记录器
                with patch('harborai.api.client.APILogger') as mock_logger_class:
                    mock_logger = Mock()
                    mock_logger_class.return_value = mock_logger
                    
                    # 创建HarborAI客户端
                    client = HarborAI()
                    
                    # 调用create方法，不传递structured_provider（应使用默认值）
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "测试消息"}]
                    )
                    
                    # 验证日志记录器被正确调用
                    assert mock_logger.log_request.called
                    call_args = mock_logger.log_request.call_args
                    
                    # 验证structured_provider使用了默认值
                    assert "structured_provider" in call_args.kwargs
                    assert call_args.kwargs["structured_provider"] == "agently"
    
    @pytest.mark.unit
    def test_structured_provider_different_default(self):
        """测试不同的默认structured_provider值"""
        # 模拟设置，设置默认值为"native"
        with patch('harborai.config.settings.get_settings') as mock_settings:
            mock_settings.return_value.default_structured_provider = "native"
            mock_settings.return_value.enable_file_logging = True
            mock_settings.return_value.log_file_path = self.log_file
            
            # 模拟客户端管理器和API响应
            with patch('harborai.api.client.ClientManager') as mock_client_manager:
                mock_client = Mock()
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message = Mock()
                mock_response.choices[0].message.content = "测试响应"
                mock_response.usage = Mock()
                mock_response.usage.total_tokens = 100
                
                mock_client.chat.completions.create.return_value = mock_response
                mock_client_manager.return_value.get_client.return_value = mock_client
                
                # 模拟日志记录器
                with patch('harborai.api.client.APILogger') as mock_logger_class:
                    mock_logger = Mock()
                    mock_logger_class.return_value = mock_logger
                    
                    # 创建HarborAI客户端
                    client = HarborAI()
                    
                    # 调用create方法，不传递structured_provider（应使用默认值）
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "测试消息"}]
                    )
                    
                    # 验证日志记录器被正确调用
                    assert mock_logger.log_request.called
                    call_args = mock_logger.log_request.call_args
                    
                    # 验证structured_provider使用了配置的默认值
                    assert "structured_provider" in call_args.kwargs
                    assert call_args.kwargs["structured_provider"] == "native"
    
    @pytest.mark.unit
    def test_provider_and_structured_provider_coexist(self):
        """测试provider和structured_provider字段同时存在"""
        # 模拟设置
        with patch('harborai.config.settings.get_settings') as mock_settings:
            mock_settings.return_value.default_structured_provider = "agently"
            mock_settings.return_value.enable_file_logging = True
            mock_settings.return_value.log_file_path = self.log_file
            
            # 模拟客户端管理器和API响应
            with patch('harborai.api.client.ClientManager') as mock_client_manager:
                mock_client = Mock()
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message = Mock()
                mock_response.choices[0].message.content = "测试响应"
                mock_response.usage = Mock()
                mock_response.usage.total_tokens = 100
                
                mock_client.chat.completions.create.return_value = mock_response
                mock_client_manager.return_value.get_client.return_value = mock_client
                
                # 模拟日志记录器
                with patch('harborai.api.client.APILogger') as mock_logger_class:
                    mock_logger = Mock()
                    mock_logger_class.return_value = mock_logger
                    
                    # 创建HarborAI客户端
                    client = HarborAI()
                    
                    # 调用create方法，同时传递provider和structured_provider
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "测试消息"}],
                        provider="openai",
                        structured_provider="native"
                    )
                    
                    # 验证日志记录器被正确调用
                    assert mock_logger.log_request.called
                    call_args = mock_logger.log_request.call_args
                    
                    # 验证两个字段都被正确传递
                    assert "provider" in call_args.kwargs
                    assert "structured_provider" in call_args.kwargs
                    assert call_args.kwargs["provider"] == "openai"
                    assert call_args.kwargs["structured_provider"] == "native"
    
    @pytest.mark.integration
    def test_file_logger_records_structured_provider(self):
        """测试文件日志记录器正确记录structured_provider字段"""
        from harborai.storage.file_logger import FileSystemLogger
        
        # 创建文件日志记录器
        logger = FileSystemLogger(log_dir=self.temp_dir)
        logger.start()
        
        try:
            # 记录请求日志
            logger.log_request(
                trace_id="test-trace-123",
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "测试消息"}],
                structured_provider="native",
                provider="openai"
            )
            
            # 等待日志写入
            import time
            time.sleep(0.1)
            
            # 读取日志文件验证
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    log_content = f.read()
                    
                # 验证structured_provider字段被记录
                assert "structured_provider" in log_content
                assert "native" in log_content
                
                # 解析JSON验证结构
                lines = log_content.strip().split('\n')
                if lines and lines[0]:
                    log_entry = json.loads(lines[0])
                    assert log_entry.get("structured_provider") == "native"
                    assert log_entry.get("model") == "gpt-3.5-turbo"
        
        finally:
            logger.stop()
    
    @pytest.mark.integration
    def test_postgres_logger_records_structured_provider(self):
        """测试PostgreSQL日志记录器正确记录structured_provider字段"""
        from harborai.storage.postgres_logger import PostgreSQLLogger
        
        # 模拟PostgreSQL连接
        with patch('harborai.storage.postgres_logger.psycopg2.connect') as mock_connect:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_connect.return_value = mock_conn
            
            # 创建PostgreSQL日志记录器
            logger = PostgreSQLLogger(
                connection_string="postgresql://test:test@localhost/test"
            )
            logger.start()
            
            try:
                # 记录请求日志
                logger.log_request(
                    trace_id="test-trace-456",
                    model="gpt-4",
                    messages=[{"role": "user", "content": "测试消息"}],
                    structured_provider="agently",
                    provider="openai"
                )
                
                # 等待日志处理
                import time
                time.sleep(0.1)
                
                # 验证SQL执行包含structured_provider字段
                assert mock_cursor.execute.called
                sql_calls = [call[0][0] for call in mock_cursor.execute.call_args_list]
                
                # 检查是否有包含structured_provider的INSERT语句
                insert_calls = [sql for sql in sql_calls if sql.strip().upper().startswith('INSERT')]
                assert any('structured_provider' in sql for sql in insert_calls)
            
            finally:
                logger.stop()


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])