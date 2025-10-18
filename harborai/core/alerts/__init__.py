#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
告警系统模块

提供完整的告警管理功能，包括：
- 告警规则管理
- 告警状态跟踪
- 通知服务
- 抑制管理
- 告警历史记录
- 命令行工具
- Web界面
"""

from datetime import datetime, timedelta

from .alert_manager import AlertManager, AlertRule, AlertSeverity, AlertCondition, AlertStatus, Alert
from .notification_service import (
    NotificationService, 
    NotificationChannel, 
    NotificationPriority,
    NotificationTemplate,
    NotificationConfig,
    NotificationRecord
)
from .suppression_manager import (
    SuppressionManager, 
    SuppressionRule, 
    SuppressionType, 
    SuppressionStatus,
    SuppressionEvent
)
from .alert_history import AlertHistory, AlertHistoryRecord, AlertEvent, AlertEventType
from .config import get_default_config, get_production_config, get_development_config
from .config_validator import ConfigValidator, validate_config_file

__all__ = [
    # 告警管理
    "AlertManager",
    "AlertRule", 
    "AlertSeverity",
    "AlertCondition",
    "AlertStatus",
    "Alert",
    
    # 通知服务
    "NotificationService",
    "NotificationChannel",
    "NotificationPriority", 
    "NotificationTemplate",
    "NotificationConfig",
    "NotificationRecord",
    
    # 抑制管理
    "SuppressionManager",
    "SuppressionRule",
    "SuppressionType",
    "SuppressionStatus", 
    "SuppressionEvent",
    
    # 历史记录
    "AlertHistory",
    "AlertHistoryRecord",
    "AlertEvent",
    "AlertEventType",
    
    # 配置
    "get_default_config",
    "get_production_config", 
    "get_development_config",
    "ConfigValidator",
    "validate_config_file",
    
    # 初始化函数
    "initialize_alert_system",
    "create_alert_manager",
    "create_notification_service",
    "create_suppression_manager",
    "create_alert_history",
    
    # 启动函数
    "start_alert_system",
    "start_web_ui",
    "run_cli"
]


async def initialize_alert_system(
    db_path: str = ":memory:",
    config: dict = None
) -> tuple[AlertManager, NotificationService, SuppressionManager, AlertHistory]:
    """
    初始化完整的告警系统
    
    Args:
        db_path: 数据库文件路径
        config: 配置字典，如果为None则使用默认配置
        
    Returns:
        tuple: (告警管理器, 通知服务, 抑制管理器, 历史记录服务)
    """
    if config is None:
        config = get_default_config()
        
    # 创建组件实例
    alert_manager = AlertManager()
    notification_service = NotificationService()
    suppression_manager = SuppressionManager()
    alert_history = AlertHistory(db_path=db_path)
    
    # 初始化组件
    await alert_manager.initialize()
    await notification_service.initialize()
    await suppression_manager.initialize()
    await alert_history.initialize()
    
    # 设置依赖关系
    alert_manager.set_notification_service(notification_service)
    alert_manager.set_suppression_service(suppression_manager)
    alert_manager.history_service = alert_history
    
    # 加载配置
    await _load_config(alert_manager, notification_service, suppression_manager, config)
    
    return alert_manager, notification_service, suppression_manager, alert_history


async def _load_config(
    alert_manager: AlertManager,
    notification_service: NotificationService, 
    suppression_manager: SuppressionManager,
    config: dict
):
    """加载配置到各个组件"""
    
    # 加载告警规则
    for rule_config in config.get("alert_rules", []):
        rule = AlertRule(
            id=rule_config["id"],
            name=rule_config["name"],
            description=rule_config["description"],
            severity=AlertSeverity(rule_config["severity"]),
            condition=AlertCondition(rule_config["condition"]),
            metric=rule_config["metric"],
            threshold=rule_config["threshold"],
            duration=timedelta(seconds=rule_config["duration"]),
            labels=rule_config.get("labels", {}),
            annotations=rule_config.get("annotations", {})
        )
        await alert_manager.add_rule(rule)
        
    # 加载通知配置
    notification_config = config.get("notification", {})
    for channel_config in notification_config.get("channels", []):
        channel = NotificationChannel(
            id=channel_config["id"],
            name=channel_config["name"],
            type=channel_config["type"],
            config=channel_config["config"],
            enabled=channel_config.get("enabled", True)
        )
        await notification_service.add_channel(channel)
        
    # 加载抑制规则
    for suppression_config in config.get("suppression_rules", []):
        rule = SuppressionRule(
            id=suppression_config["id"],
            name=suppression_config["name"],
            type=SuppressionType(suppression_config["type"]),
            target_labels=suppression_config["target_labels"],
            start_time=datetime.fromisoformat(suppression_config["start_time"]) if suppression_config.get("start_time") else None,
            end_time=datetime.fromisoformat(suppression_config["end_time"]) if suppression_config.get("end_time") else None,
            reason=suppression_config.get("reason", "配置加载")
        )
        await suppression_manager.add_suppression_rule(rule)


async def create_alert_manager(config: dict = None) -> AlertManager:
    """创建告警管理器"""
    manager = AlertManager()
    await manager.initialize()
    
    if config:
        for rule_config in config.get("alert_rules", []):
            rule = AlertRule(
                id=rule_config["id"],
                name=rule_config["name"],
                description=rule_config["description"],
                severity=AlertSeverity(rule_config["severity"]),
                condition=AlertCondition(rule_config["condition"]),
                metric=rule_config["metric"],
                threshold=rule_config["threshold"],
                duration=timedelta(seconds=rule_config["duration"]),
                labels=rule_config.get("labels", {}),
                annotations=rule_config.get("annotations", {})
            )
            await manager.add_rule(rule)
            
    return manager


async def create_notification_service(config: dict = None) -> NotificationService:
    """创建通知服务"""
    service = NotificationService()
    await service.initialize()
    
    if config:
        notification_config = config.get("notification", {})
        for channel_config in notification_config.get("channels", []):
            channel = NotificationChannel(
                id=channel_config["id"],
                name=channel_config["name"],
                type=channel_config["type"],
                config=channel_config["config"],
                enabled=channel_config.get("enabled", True)
            )
            await service.add_channel(channel)
            
    return service


async def create_suppression_manager(config: dict = None) -> SuppressionManager:
    """创建抑制管理器"""
    manager = SuppressionManager()
    await manager.initialize()
    
    if config:
        for suppression_config in config.get("suppression_rules", []):
            rule = SuppressionRule(
                id=suppression_config["id"],
                name=suppression_config["name"],
                type=SuppressionType(suppression_config["type"]),
                target_labels=suppression_config["target_labels"],
                start_time=datetime.fromisoformat(suppression_config["start_time"]) if suppression_config.get("start_time") else None,
                end_time=datetime.fromisoformat(suppression_config["end_time"]) if suppression_config.get("end_time") else None,
                reason=suppression_config.get("reason", "配置加载")
            )
            await manager.add_suppression_rule(rule)
            
    return manager


async def create_alert_history(db_path: str = ":memory:") -> AlertHistory:
    """创建告警历史服务"""
    history = AlertHistory(db_path=db_path)
    await history.initialize()
    return history


async def start_alert_system(
    db_path: str = "alerts.db",
    config_path: str = None,
    enable_web_ui: bool = False,
    web_port: int = 8080
) -> tuple[AlertManager, NotificationService, SuppressionManager, AlertHistory]:
    """
    启动完整的告警系统
    
    Args:
        db_path: 数据库文件路径
        config_path: 配置文件路径，如果为None则使用默认配置
        enable_web_ui: 是否启用Web界面
        web_port: Web界面端口
        
    Returns:
        tuple: (告警管理器, 通知服务, 抑制管理器, 历史记录服务)
    """
    import json
    
    # 加载配置
    if config_path:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = get_default_config()
        
    # 验证配置
    is_valid, results = validate_config_file(config_path) if config_path else (True, [])
    if not is_valid:
        print("配置验证失败:")
        validator = ConfigValidator()
        validator.results = results
        print(validator.format_results("text"))
        raise ValueError("配置文件验证失败")
        
    # 初始化系统
    components = await initialize_alert_system(db_path=db_path, config=config)
    alert_manager, notification_service, suppression_manager, alert_history = components
    
    # 启动告警管理器
    await alert_manager.start()
    
    # 启动Web界面
    if enable_web_ui:
        await start_web_ui(db_path=db_path, port=web_port)
        
    print(f"告警系统已启动")
    print(f"数据库: {db_path}")
    if enable_web_ui:
        print(f"Web界面: http://localhost:{web_port}")
        
    return components


async def start_web_ui(db_path: str = ":memory:", port: int = 8080):
    """启动Web界面"""
    from .web_ui import create_app
    import uvicorn
    
    app = await create_app(db_path=db_path)
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)
    
    print(f"启动Web界面: http://localhost:{port}")
    await server.serve()


def run_cli():
    """运行命令行工具"""
    from .cli import main
    import asyncio
    import sys
    
    try:
        result = asyncio.run(main())
        sys.exit(result)
    except KeyboardInterrupt:
        print("\n操作被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"执行命令时发生错误: {e}")
        sys.exit(1)