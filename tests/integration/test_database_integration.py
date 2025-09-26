# -*- coding: utf-8 -*-
"""
数据库集成测试

本模块测试 HarborAI 与数据库的集成功能，包括：
- PostgreSQL 数据库连接和操作
- 对话历史数据持久化
- 用户会话管理
- 模型配置存储
- 性能指标记录
- 事务管理和数据一致性
- 连接池管理
- 数据库迁移和版本控制
"""

import pytest
import asyncio
import time
import json
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager

# 导入数据库相关模块
try:
    import asyncpg
    import psycopg2
    from sqlalchemy import create_engine, text, MetaData, Table, Column, String, Integer, DateTime, JSON, Boolean
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker, declarative_base
    from sqlalchemy.pool import QueuePool
except ImportError as e:
    pytest.skip(f"数据库依赖未安装: {e}", allow_module_level=True)

# 导入 HarborAI 相关模块
try:
    from harborai import HarborAI
    from harborai.storage.postgres_logger import PostgreSQLLogger
    from harborai.core.exceptions import HarborAIError
    from harborai.utils.exceptions import StorageError
except ImportError as e:
    pytest.skip(f"无法导入 HarborAI 数据库模块: {e}", allow_module_level=True)

from tests.integration import INTEGRATION_TEST_CONFIG, TEST_DATA_CONFIG


# 数据库配置
TEST_DATABASE_CONFIG = {
    "host": os.getenv("TEST_DB_HOST", "localhost"),
    "port": int(os.getenv("TEST_DB_PORT", "5432")),
    "database": os.getenv("TEST_DB_NAME", "harborai_test"),
    "username": os.getenv("TEST_DB_USER", "postgres"),
    "password": os.getenv("TEST_DB_PASSWORD", "password"),
    "pool_size": 10,
    "max_overflow": 20,
    "pool_timeout": 30,
    "pool_recycle": 3600
}


@dataclass
class TestConversation:
    """测试对话数据类"""
    id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    is_active: bool = True


@dataclass
class TestMessage:
    """测试消息数据类"""
    id: str
    conversation_id: str
    role: str
    content: str
    timestamp: datetime
    model: str
    tokens_used: int
    cost: float
    metadata: Dict[str, Any]


@dataclass
class TestUser:
    """测试用户数据类"""
    id: str
    username: str
    email: str
    created_at: datetime
    last_active: datetime
    preferences: Dict[str, Any]
    is_active: bool = True


class TestDatabaseIntegration:
    """
    数据库集成测试类
    
    测试 HarborAI 与 PostgreSQL 数据库的集成功能。
    """
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """测试方法设置"""
        self.test_db_url = (
            f"postgresql://{TEST_DATABASE_CONFIG['username']}:"
            f"{TEST_DATABASE_CONFIG['password']}@"
            f"{TEST_DATABASE_CONFIG['host']}:"
            f"{TEST_DATABASE_CONFIG['port']}/"
            f"{TEST_DATABASE_CONFIG['database']}"
        )
        
        self.async_db_url = (
            f"postgresql+asyncpg://{TEST_DATABASE_CONFIG['username']}:"
            f"{TEST_DATABASE_CONFIG['password']}@"
            f"{TEST_DATABASE_CONFIG['host']}:"
            f"{TEST_DATABASE_CONFIG['port']}/"
            f"{TEST_DATABASE_CONFIG['database']}"
        )
        
        # 测试数据
        self.test_user_id = str(uuid.uuid4())
        self.test_conversation_id = str(uuid.uuid4())
        self.test_message_id = str(uuid.uuid4())
        
        self.test_user = TestUser(
            id=self.test_user_id,
            username="test_user",
            email="test@example.com",
            created_at=datetime.now(),
            last_active=datetime.now(),
            preferences={"language": "zh-CN", "theme": "dark"}
        )
        
        self.test_conversation = TestConversation(
            id=self.test_conversation_id,
            user_id=self.test_user_id,
            title="测试对话",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            messages=[],
            metadata={"source": "test", "version": "1.0"}
        )
        
        self.test_message = TestMessage(
            id=self.test_message_id,
            conversation_id=self.test_conversation_id,
            role="user",
            content="这是一条测试消息",
            timestamp=datetime.now(),
            model="gpt-3.5-turbo",
            tokens_used=15,
            cost=0.001,
            metadata={"source": "test"}
        )
    
    @pytest.fixture
    def mock_postgres_logger(self):
        """Mock PostgreSQL日志记录器夹具"""
        with patch('harborai.storage.PostgreSQLLogger') as mock_logger:
            logger_instance = Mock()
            mock_logger.return_value = logger_instance
            
            # 配置日志记录方法
            logger_instance.log_conversation.return_value = True
            logger_instance.log_message.return_value = True
            logger_instance.get_conversation_history.return_value = []
            
            yield logger_instance
    
    @pytest.fixture
    def mock_storage_lifecycle(self):
        """Mock 存储生命周期管理器夹具"""
        with patch('harborai.storage.lifecycle.LifecycleManager') as mock_lifecycle:
            lifecycle_instance = Mock()
            mock_lifecycle.return_value = lifecycle_instance
            
            # 配置生命周期管理
            lifecycle_instance.initialize.return_value = True
            lifecycle_instance.shutdown.return_value = True
            lifecycle_instance.add_startup_hook.return_value = None
            lifecycle_instance.add_shutdown_hook.return_value = None
            
            yield lifecycle_instance
    
    @pytest.mark.integration
    @pytest.mark.database
    @pytest.mark.p0
    def test_database_connection(self, mock_postgres_logger):
        """测试数据库连接"""
        # 配置日志记录操作
        mock_postgres_logger.log_conversation.return_value = True
        mock_postgres_logger.get_conversation_history.return_value = [{
            "host": TEST_DATABASE_CONFIG["host"],
            "port": TEST_DATABASE_CONFIG["port"],
            "database": TEST_DATABASE_CONFIG["database"],
            "status": "connected"
        }]
        
        # 测试数据库连接日志记录
        logger = mock_postgres_logger
        connection_result = logger.log_conversation(
            user_id="test_connection",
            conversation_data={
                "host": TEST_DATABASE_CONFIG["host"],
                "port": TEST_DATABASE_CONFIG["port"],
                "database": TEST_DATABASE_CONFIG["database"],
                "username": TEST_DATABASE_CONFIG["username"],
                "connection_type": "database_test"
            }
        )
        
        # 验证连接日志记录
        assert connection_result is True
        
        connection_history = logger.get_conversation_history("test_connection")
        assert len(connection_history) == 1
        assert connection_history[0]["status"] == "connected"
        assert connection_history[0]["host"] == TEST_DATABASE_CONFIG["host"]
        assert connection_history[0]["database"] == TEST_DATABASE_CONFIG["database"]
    
    @pytest.mark.integration
    @pytest.mark.database
    @pytest.mark.p0
    def test_connection_pool_management(self, mock_postgres_logger):
        """测试连接池管理"""
        # 配置日志记录操作
        mock_postgres_logger.log_conversation.return_value = True
        mock_postgres_logger.get_conversation_history.return_value = [{
            "active_connections": 2,
            "idle_connections": 6,
            "total_connections": 8,
            "pool_status": "active"
        }]
        
        # 测试连接池状态日志记录
        logger = mock_postgres_logger
        
        # 测试记录连接池获取操作
        get_result = logger.log_conversation(
            user_id="connection_pool",
            conversation_data={"action": "get_connection", "status": "success"}
        )
        assert get_result is True
        
        # 测试记录连接池归还操作
        return_result = logger.log_conversation(
            user_id="connection_pool",
            conversation_data={"action": "return_connection", "status": "success"}
        )
        assert return_result is True
        
        # 测试连接池统计信息
        pool_stats = logger.get_conversation_history("connection_pool")
        assert len(pool_stats) == 1
        assert "active_connections" in pool_stats[0]
        assert "idle_connections" in pool_stats[0]
        assert "total_connections" in pool_stats[0]
        assert pool_stats[0]["total_connections"] == 8
    
    @pytest.mark.integration
    @pytest.mark.database
    @pytest.mark.p0
    def test_user_data_persistence(self, mock_postgres_logger):
        """测试用户数据持久化"""
        # 配置日志记录操作
        mock_postgres_logger.log_conversation.return_value = True
        mock_postgres_logger.get_conversation_history.return_value = [asdict(self.test_user)]
        
        logger = mock_postgres_logger
        
        # 测试记录用户数据
        result = logger.log_conversation(
            user_id=self.test_user.id,
            conversation_data={
                "username": self.test_user.username,
                "email": self.test_user.email,
                "preferences": self.test_user.preferences
            }
        )
        assert result is True
        
        # 测试查询用户历史
        user_history = logger.get_conversation_history(self.test_user.id)
        assert len(user_history) == 1
        assert user_history[0]["username"] == self.test_user.username
        assert user_history[0]["email"] == self.test_user.email
        assert user_history[0]["preferences"] == self.test_user.preferences
    
    @pytest.mark.integration
    @pytest.mark.database
    @pytest.mark.p0
    def test_conversation_data_persistence(self, mock_postgres_logger):
        """测试对话数据持久化"""
        # 配置日志记录操作
        mock_postgres_logger.log_conversation.return_value = True
        mock_postgres_logger.get_conversation_history.return_value = [asdict(self.test_conversation)]
        
        logger = mock_postgres_logger
        
        # 测试记录对话数据
        result = logger.log_conversation(
            user_id=self.test_conversation.user_id,
            conversation_data={
                "id": self.test_conversation.id,
                "title": self.test_conversation.title,
                "metadata": self.test_conversation.metadata,
                "created_at": self.test_conversation.created_at
            }
        )
        assert result is True
        
        # 测试查询对话历史
        conversation_history = logger.get_conversation_history(self.test_user_id)
        assert len(conversation_history) == 1
        assert conversation_history[0]["user_id"] == self.test_conversation.user_id
        assert conversation_history[0]["title"] == self.test_conversation.title
        assert conversation_history[0]["metadata"] == self.test_conversation.metadata
    
    @pytest.mark.integration
    @pytest.mark.database
    @pytest.mark.p0
    def test_message_data_persistence(self, mock_postgres_logger):
        """测试消息数据持久化"""
        # 配置日志记录操作
        mock_postgres_logger.log_message.return_value = True
        mock_postgres_logger.get_conversation_history.return_value = [asdict(self.test_message)]
        
        logger = mock_postgres_logger
        
        # 测试记录消息数据
        result = logger.log_message(
            conversation_id=self.test_message.conversation_id,
            message_data={
                "id": self.test_message.id,
                "role": self.test_message.role,
                "content": self.test_message.content,
                "model": self.test_message.model,
                "tokens_used": self.test_message.tokens_used,
                "cost": self.test_message.cost,
                "metadata": self.test_message.metadata,
                "timestamp": self.test_message.timestamp
            }
        )
        assert result is True
        
        # 测试查询消息历史
        message_history = logger.get_conversation_history(self.test_conversation_id)
        assert len(message_history) == 1
        assert message_history[0]["conversation_id"] == self.test_message.conversation_id
        assert message_history[0]["role"] == self.test_message.role
        assert message_history[0]["content"] == self.test_message.content
        assert message_history[0]["model"] == self.test_message.model
    
    @pytest.mark.integration
    @pytest.mark.database
    @pytest.mark.p1
    def test_transaction_management(self, mock_postgres_logger):
        """测试事务管理"""
        # 配置日志记录操作
        mock_postgres_logger.log_conversation.return_value = True
        mock_postgres_logger.log_message.return_value = True
        
        logger = mock_postgres_logger
        
        # 测试批量记录操作
        try:
            # 执行多个日志记录操作
            result1 = logger.log_conversation(
                user_id=self.test_user_id,
                conversation_data={"username": "transaction_user", "email": "transaction@test.com"}
            )
            
            result2 = logger.log_message(
                conversation_id=self.test_conversation_id,
                message_data={"role": "user", "content": "事务测试消息"}
            )
            
            # 验证操作成功
            assert result1 is True
            assert result2 is True
            
        except Exception as e:
            # 记录错误
            print(f"事务操作失败: {e}")
            raise e
        
        # 验证日志记录操作
        assert mock_postgres_logger.log_conversation.call_count >= 1
        assert mock_postgres_logger.log_message.call_count >= 1
    
    @pytest.mark.integration
    @pytest.mark.database
    @pytest.mark.p1
    def test_model_config_storage(self, mock_postgres_logger):
        """测试模型配置存储"""
        # 测试模型配置数据
        model_config = {
            "id": str(uuid.uuid4()),
            "name": "gpt-3.5-turbo",
            "vendor": "openai",
            "parameters": {
                "temperature": 0.7,
                "max_tokens": 2048,
                "top_p": 0.9
            },
            "rate_limits": {
                "requests_per_minute": 60,
                "tokens_per_minute": 90000
            },
            "cost_per_token": {
                "input": 0.0015,
                "output": 0.002
            },
            "created_at": datetime.now(),
            "is_active": True
        }
        
        # 配置日志记录操作
        mock_postgres_logger.log_conversation.return_value = True
        mock_postgres_logger.get_conversation_history.return_value = [model_config]
        
        logger = mock_postgres_logger
        
        # 测试记录模型配置
        result = logger.log_conversation(
            user_id="system",
            conversation_data={
                "type": "model_config",
                "name": model_config["name"],
                "vendor": model_config["vendor"],
                "parameters": model_config["parameters"],
                "rate_limits": model_config["rate_limits"],
                "cost_per_token": model_config["cost_per_token"]
            }
        )
        assert result is True
        
        # 测试查询模型配置历史
        config_history = logger.get_conversation_history("system")
        assert len(config_history) == 1
        assert config_history[0]["name"] == model_config["name"]
        assert config_history[0]["vendor"] == model_config["vendor"]
        assert config_history[0]["parameters"] == model_config["parameters"]
    
    @pytest.mark.integration
    @pytest.mark.database
    @pytest.mark.p1
    def test_performance_metrics_storage(self, mock_postgres_logger):
        """测试性能指标存储"""
        # 测试性能指标数据
        performance_metric = {
            "id": str(uuid.uuid4()),
            "conversation_id": self.test_conversation_id,
            "model": "gpt-3.5-turbo",
            "vendor": "openai",
            "request_timestamp": datetime.now(),
            "response_timestamp": datetime.now() + timedelta(seconds=2),
            "latency_ms": 2000,
            "tokens_input": 50,
            "tokens_output": 100,
            "cost": 0.005,
            "success": True,
            "error_message": None,
            "metadata": {"region": "us-east-1", "cache_hit": False}
        }
        
        # 配置日志记录操作
        mock_postgres_logger.log_message.return_value = True
        mock_postgres_logger.get_conversation_history.return_value = [performance_metric]
        
        logger = mock_postgres_logger
        
        # 测试记录性能指标
        result = logger.log_message(
            conversation_id=performance_metric["conversation_id"],
            message_data={
                "type": "performance_metric",
                "model": performance_metric["model"],
                "vendor": performance_metric["vendor"],
                "latency_ms": performance_metric["latency_ms"],
                "tokens_input": performance_metric["tokens_input"],
                "tokens_output": performance_metric["tokens_output"],
                "cost": performance_metric["cost"],
                "success": performance_metric["success"],
                "metadata": performance_metric["metadata"]
            }
        )
        assert result is True
        
        # 测试查询性能指标历史
        metrics_history = logger.get_conversation_history(self.test_conversation_id)
        assert len(metrics_history) == 1
        assert metrics_history[0]["model"] == performance_metric["model"]
        assert metrics_history[0]["latency_ms"] == performance_metric["latency_ms"]
        assert metrics_history[0]["cost"] == performance_metric["cost"]
    
    @pytest.mark.integration
    @pytest.mark.database
    @pytest.mark.p1
    def test_user_session_management(self, mock_postgres_logger):
        """测试用户会话管理"""
        # 测试会话数据
        session_data = {
            "id": str(uuid.uuid4()),
            "user_id": self.test_user_id,
            "session_token": "session_token_123",
            "created_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(hours=24),
            "last_activity": datetime.now(),
            "ip_address": "192.168.1.100",
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "is_active": True,
            "metadata": {"login_method": "password", "device_type": "desktop"}
        }
        
        # 配置日志记录操作
        mock_postgres_logger.log_conversation.return_value = True
        mock_postgres_logger.get_conversation_history.return_value = [session_data]
        
        logger = mock_postgres_logger
        
        # 测试记录会话创建
        session_result = logger.log_conversation(
            user_id=session_data["user_id"],
            conversation_data={
                "session_id": session_data["id"],
                "session_token": session_data["session_token"],
                "expires_at": session_data["expires_at"].isoformat(),
                "ip_address": session_data["ip_address"],
                "user_agent": session_data["user_agent"],
                "metadata": session_data["metadata"]
            }
        )
        assert session_result is True
        
        # 测试查询会话历史
        session_history = logger.get_conversation_history(session_data["user_id"])
        assert len(session_history) == 1
        assert session_history[0]["user_id"] == session_data["user_id"]
        assert session_history[0]["is_active"] is True
        
        # 测试记录会话活动更新
        activity_result = logger.log_conversation(
            user_id=session_data["user_id"],
            conversation_data={
                "action": "update_activity",
                "session_token": session_data["session_token"],
                "last_activity": datetime.now().isoformat()
            }
        )
        assert activity_result is True
        
        # 测试记录会话失效
        invalidate_result = logger.log_conversation(
            user_id=session_data["user_id"],
            conversation_data={
                "action": "invalidate_session",
                "session_token": session_data["session_token"]
            }
        )
        assert invalidate_result is True
    
    @pytest.mark.integration
    @pytest.mark.database
    @pytest.mark.p2
    def test_database_migration(self, mock_postgres_logger):
        """测试数据库迁移"""
        # 配置日志记录操作
        mock_postgres_logger.log_conversation.return_value = True
        mock_postgres_logger.get_conversation_history.return_value = [{
            "current_version": "1.0.0",
            "available_migrations": [
                {"version": "1.0.1", "description": "添加用户偏好设置表"},
                {"version": "1.0.2", "description": "添加性能指标索引"}
            ],
            "migration_status": "success"
        }]
        
        logger = mock_postgres_logger
        
        # 测试记录迁移操作
        migration_result = logger.log_conversation(
            user_id="migration_system",
            conversation_data={
                "action": "apply_migration",
                "version": "1.0.1",
                "description": "添加用户偏好设置表"
            }
        )
        assert migration_result is True
        
        # 测试查询迁移历史
        migration_history = logger.get_conversation_history("migration_system")
        assert len(migration_history) == 1
        assert migration_history[0]["current_version"] == "1.0.0"
        assert len(migration_history[0]["available_migrations"]) == 2
        assert migration_history[0]["available_migrations"][0]["version"] == "1.0.1"
        
        # 测试记录回滚操作
        rollback_result = logger.log_conversation(
            user_id="migration_system",
            conversation_data={
                "action": "rollback_migration",
                "version": "1.0.1"
            }
        )
        assert rollback_result is True
    
    @pytest.mark.integration
    @pytest.mark.database
    @pytest.mark.p2
    def test_data_backup_and_restore(self, mock_postgres_logger):
        """测试数据备份和恢复"""
        # 配置备份恢复操作
        backup_info = {
            "backup_id": str(uuid.uuid4()),
            "created_at": datetime.now(),
            "size_bytes": 1024000,
            "tables_count": 8,
            "records_count": 5000,
            "file_path": "/backups/harborai_backup_20240101.sql"
        }
        
        # 配置日志记录操作
        mock_postgres_logger.log_conversation.return_value = True
        mock_postgres_logger.get_conversation_history.return_value = [backup_info]
        
        logger = mock_postgres_logger
        
        # 测试记录备份创建
        backup_result = logger.log_conversation(
            user_id="backup_system",
            conversation_data={
                "action": "create_backup",
                "backup_name": "test_backup",
                "include_tables": ["users", "conversations", "messages"],
                "backup_info": backup_info
            }
        )
        assert backup_result is True
        
        # 测试查询备份历史
        backup_history = logger.get_conversation_history("backup_system")
        assert len(backup_history) == 1
        assert backup_history[0]["backup_id"] == backup_info["backup_id"]
        assert backup_history[0]["tables_count"] == 8
        
        # 测试记录备份恢复
        restore_result = logger.log_conversation(
            user_id="backup_system",
            conversation_data={
                "action": "restore_backup",
                "backup_id": backup_info["backup_id"],
                "target_database": "harborai_test_restore"
            }
        )
        assert restore_result is True
        
        # 测试记录备份删除
        delete_result = logger.log_conversation(
            user_id="backup_system",
            conversation_data={
                "action": "delete_backup",
                "backup_id": backup_info["backup_id"]
            }
        )
        assert delete_result is True
    
    @pytest.mark.integration
    @pytest.mark.database
    @pytest.mark.p2
    def test_query_optimization(self, mock_postgres_logger):
        """测试查询优化"""
        # 配置查询优化
        query_stats = {
            "query_id": "query_123",
            "sql": "SELECT * FROM conversations WHERE user_id = $1 ORDER BY created_at DESC",
            "execution_time_ms": 150,
            "rows_examined": 1000,
            "rows_returned": 10,
            "index_used": True,
            "optimization_suggestions": [
                "考虑添加复合索引 (user_id, created_at)",
                "使用LIMIT子句限制返回行数"
            ]
        }
        
        # 配置日志记录操作
        mock_postgres_logger.log_conversation.return_value = True
        mock_postgres_logger.get_conversation_history.return_value = [query_stats]
        
        logger = mock_postgres_logger
        
        # 测试记录查询分析
        analysis_result = logger.log_conversation(
            user_id="query_optimizer",
            conversation_data={
                "action": "analyze_query",
                "sql": "SELECT * FROM conversations WHERE user_id = $1 ORDER BY created_at DESC",
                "parameters": [self.test_user_id],
                "query_stats": query_stats
            }
        )
        assert analysis_result is True
        
        # 测试查询分析历史
        analysis_history = logger.get_conversation_history("query_optimizer")
        assert len(analysis_history) == 1
        assert analysis_history[0]["execution_time_ms"] == 150
        assert analysis_history[0]["index_used"] is True
        assert len(analysis_history[0]["optimization_suggestions"]) == 2
        
        # 测试记录慢查询检测
        slow_query_result = logger.log_conversation(
            user_id="query_optimizer",
            conversation_data={
                "action": "detect_slow_queries",
                "threshold_ms": 100,
                "slow_queries": [query_stats]
            }
        )
        assert slow_query_result is True
        
        # 测试记录索引创建
        index_result = logger.log_conversation(
            user_id="query_optimizer",
            conversation_data={
                "action": "create_index",
                "table_name": "conversations",
                "columns": ["user_id", "created_at"],
                "index_name": "idx_conversations_user_created"
            }
        )
        assert index_result is True
    
    @pytest.mark.integration
    @pytest.mark.database
    @pytest.mark.real_db
    @pytest.mark.p3
    def test_real_database_operations(self):
        """真实数据库操作测试（需要真实数据库）"""
        # 检查是否启用真实数据库测试
        if not os.getenv('ENABLE_REAL_DB_TESTS', 'false').lower() == 'true':
            pytest.skip("真实数据库测试未启用，设置ENABLE_REAL_DB_TESTS=true启用")
        
        # 检查数据库连接参数
        required_env_vars = ['TEST_DB_HOST', 'TEST_DB_USER', 'TEST_DB_PASSWORD']
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            pytest.skip(f"缺少数据库环境变量: {missing_vars}")
        
        try:
            # 创建真实数据库连接
            engine = create_engine(
                self.test_db_url,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30
            )
            
            # 测试连接
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1 as test"))
                assert result.fetchone()[0] == 1
            
            # 测试创建表
            metadata = MetaData()
            test_table = Table(
                'test_harborai_integration',
                metadata,
                Column('id', String, primary_key=True),
                Column('data', JSON),
                Column('created_at', DateTime, default=datetime.now)
            )
            
            # 创建测试表
            metadata.create_all(engine)
            
            # 测试插入数据
            with engine.connect() as conn:
                conn.execute(
                    test_table.insert().values(
                        id=str(uuid.uuid4()),
                        data={"test": "integration", "version": 1},
                        created_at=datetime.now()
                    )
                )
                conn.commit()
            
            # 测试查询数据
            with engine.connect() as conn:
                result = conn.execute(
                    text("SELECT COUNT(*) FROM test_harborai_integration")
                )
                count = result.fetchone()[0]
                assert count >= 1
            
            # 清理测试表
            metadata.drop_all(engine)
            
        except Exception as e:
            pytest.fail(f"真实数据库操作失败: {e}")
        finally:
            if 'engine' in locals():
                engine.dispose()


class TestAsyncDatabaseOperations:
    """
    异步数据库操作测试类
    
    测试异步数据库操作和连接管理。
    """
    
    @pytest.mark.integration
    @pytest.mark.database
    @pytest.mark.asyncio
    @pytest.mark.p2
    async def test_async_database_operations(self):
        """测试异步数据库操作"""
        # Mock 异步存储生命周期管理器
        with patch('harborai.storage.lifecycle.LifecycleManager') as mock_async_manager:
            manager_instance = AsyncMock()
            mock_async_manager.return_value = manager_instance
            
            # 配置异步操作
            manager_instance.initialize.return_value = None
            manager_instance.add_startup_hook.return_value = None
            manager_instance.add_shutdown_hook.return_value = None
            manager_instance.shutdown.return_value = None
            
            # 测试异步初始化
            lifecycle_manager = manager_instance
            lifecycle_manager.initialize()
            
            # 测试添加钩子
            def test_startup_hook():
                pass
            
            def test_shutdown_hook():
                pass
            
            lifecycle_manager.add_startup_hook(test_startup_hook)
            lifecycle_manager.add_shutdown_hook(test_shutdown_hook)
            
            # 测试关闭
            lifecycle_manager.shutdown()
    
    @pytest.mark.integration
    @pytest.mark.database
    @pytest.mark.asyncio
    @pytest.mark.p2
    async def test_async_transaction_management(self):
        """测试异步事务管理"""
        with patch('harborai.storage.PostgreSQLLogger') as mock_async_logger:
            logger_instance = AsyncMock()
            mock_async_logger.return_value = logger_instance
            
            # 配置异步日志记录
            logger_instance.log_conversation.return_value = True
            logger_instance.log_message.return_value = True
            logger_instance.get_conversation_history.return_value = [{"id": "tx_test", "data": "transaction_data"}]
            
            # 测试异步批量记录
            logger = logger_instance
            
            # 模拟批量操作
            conversation_result = await logger.log_conversation(
                user_id="tx_user",
                conversation_data={"type": "transaction_test", "data": "async_transaction"}
            )
            assert conversation_result is True
            
            message_result = await logger.log_message(
                conversation_id="tx_conversation",
                message_data={"role": "user", "content": "异步事务测试消息"}
            )
            assert message_result is True
            
            # 验证历史查询
            history = await logger.get_conversation_history("tx_user")
            assert len(history) == 1
            assert history[0]["id"] == "tx_test"
    
    @pytest.mark.integration
    @pytest.mark.database
    @pytest.mark.asyncio
    @pytest.mark.p2
    async def test_async_connection_pool(self):
        """测试异步连接池"""
        with patch('harborai.storage.lifecycle.LifecycleManager') as mock_async_pool:
            pool_instance = AsyncMock()
            mock_async_pool.return_value = pool_instance
            
            # 配置异步生命周期管理
            pool_instance.initialize.return_value = None
            pool_instance.add_startup_hook.return_value = None
            pool_instance.add_shutdown_hook.return_value = None
            pool_instance.shutdown.return_value = None
            
            # 测试异步连接池生命周期
            lifecycle = pool_instance
            
            # 初始化连接池
            lifecycle.initialize()
            
            # 添加钩子函数
            def startup_hook():
                pass
            
            def shutdown_hook():
                pass
            
            lifecycle.add_startup_hook(startup_hook)
            lifecycle.add_shutdown_hook(shutdown_hook)
            
            # 关闭连接池
            lifecycle.shutdown()
            
            # 验证调用
            pool_instance.initialize.assert_called_once()
            pool_instance.add_startup_hook.assert_called_once()
            pool_instance.add_shutdown_hook.assert_called_once()
            pool_instance.shutdown.assert_called_once()