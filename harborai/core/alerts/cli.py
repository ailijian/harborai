#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
告警系统命令行工具

提供告警系统的命令行管理接口
"""

import asyncio
import json
import sys
import argparse
import tempfile
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

from .alert_manager import AlertManager, AlertRule, AlertSeverity, AlertCondition
from .notification_service import NotificationService
from .suppression_manager import SuppressionManager, SuppressionRule, SuppressionType
from .alert_history import AlertHistory
from .config import get_default_config, get_production_config, get_development_config
from .config_validator import ConfigValidator, validate_config_file


class AlertCLI:
    """告警系统命令行接口"""
    
    def __init__(self):
        self.alert_manager = None
        self.notification_service = None
        self.suppression_manager = None
        self.alert_history = None
        
    async def initialize(self, db_path: Optional[str] = None):
        """初始化组件"""
        if db_path is None:
            # 使用临时数据库
            fd, db_path = tempfile.mkstemp(suffix='.db')
            os.close(fd)
            
        self.alert_manager = AlertManager()
        self.notification_service = NotificationService()
        self.suppression_manager = SuppressionManager()
        self.alert_history = AlertHistory(db_path=db_path)
        
        # 初始化组件
        await self.alert_manager.initialize()
        await self.notification_service.initialize()
        await self.suppression_manager.initialize()
        await self.alert_history.initialize()
        
        # 设置依赖关系
        self.alert_manager.set_notification_service(self.notification_service)
        self.alert_manager.set_suppression_service(self.suppression_manager)
        self.alert_manager.history_service = self.alert_history
        
    async def cleanup(self):
        """清理资源"""
        if self.alert_manager:
            await self.alert_manager.stop()
            
    # 配置管理命令
    async def cmd_validate_config(self, args):
        """验证配置文件"""
        print(f"验证配置文件: {args.config}")
        
        is_valid, results = validate_config_file(args.config)
        
        validator = ConfigValidator()
        validator.results = results
        
        if args.format == "json":
            print(validator.format_results("json"))
        else:
            print(validator.format_results("text"))
            
        return 0 if is_valid else 1
        
    async def cmd_generate_config(self, args):
        """生成配置文件"""
        if args.type == "default":
            config = get_default_config()
        elif args.type == "production":
            config = get_production_config()
        elif args.type == "development":
            config = get_development_config()
        else:
            print(f"错误: 不支持的配置类型 '{args.type}'")
            return 1
            
        config_json = json.dumps(config, indent=2, ensure_ascii=False)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(config_json)
            print(f"配置已保存到: {args.output}")
        else:
            print(config_json)
            
        return 0
        
    # 规则管理命令
    async def cmd_list_rules(self, args):
        """列出告警规则"""
        await self.initialize(args.db)
        
        try:
            rules = await self.alert_manager.get_rules()
            
            if args.format == "json":
                rules_data = []
                for rule in rules:
                    rules_data.append({
                        "id": rule.id,
                        "name": rule.name,
                        "description": rule.description,
                        "severity": rule.severity.value,
                        "condition": rule.condition.value,
                        "metric": rule.metric,
                        "threshold": rule.threshold,
                        "duration": rule.duration.total_seconds(),
                        "enabled": rule.enabled,
                        "labels": rule.labels,
                        "annotations": rule.annotations
                    })
                print(json.dumps(rules_data, indent=2, ensure_ascii=False))
            else:
                print(f"{'ID':<20} {'名称':<30} {'严重级别':<10} {'状态':<10}")
                print("-" * 80)
                for rule in rules:
                    status = "启用" if rule.enabled else "禁用"
                    print(f"{rule.id:<20} {rule.name:<30} {rule.severity.value:<10} {status:<10}")
                    
        finally:
            await self.cleanup()
            
        return 0
        
    async def cmd_add_rule(self, args):
        """添加告警规则"""
        await self.initialize(args.db)
        
        try:
            # 从文件或命令行参数创建规则
            if args.file:
                with open(args.file, 'r', encoding='utf-8') as f:
                    rule_data = json.load(f)
            else:
                rule_data = {
                    "id": args.id,
                    "name": args.name,
                    "description": args.description or args.name,
                    "severity": args.severity,
                    "condition": args.condition,
                    "metric": args.metric,
                    "threshold": args.threshold,
                    "duration": args.duration,
                    "labels": json.loads(args.labels) if args.labels else {},
                    "annotations": json.loads(args.annotations) if args.annotations else {}
                }
                
            # 创建规则对象
            rule = AlertRule(
                id=rule_data["id"],
                name=rule_data["name"],
                description=rule_data["description"],
                severity=AlertSeverity(rule_data["severity"]),
                condition=AlertCondition(rule_data["condition"]),
                metric=rule_data["metric"],
                threshold=rule_data["threshold"],
                duration=timedelta(seconds=rule_data["duration"]),
                labels=rule_data.get("labels", {}),
                annotations=rule_data.get("annotations", {})
            )
            
            await self.alert_manager.add_rule(rule)
            print(f"成功添加告警规则: {rule.id}")
            
        except Exception as e:
            print(f"添加规则失败: {e}")
            return 1
        finally:
            await self.cleanup()
            
        return 0
        
    async def cmd_remove_rule(self, args):
        """删除告警规则"""
        await self.initialize(args.db)
        
        try:
            await self.alert_manager.remove_rule(args.id)
            print(f"成功删除告警规则: {args.id}")
            
        except Exception as e:
            print(f"删除规则失败: {e}")
            return 1
        finally:
            await self.cleanup()
            
        return 0
        
    # 告警管理命令
    async def cmd_list_alerts(self, args):
        """列出活跃告警"""
        await self.initialize(args.db)
        
        try:
            if args.severity:
                alerts = await self.alert_manager.get_alerts_by_severity(
                    AlertSeverity(args.severity)
                )
            else:
                alerts = await self.alert_manager.get_active_alerts()
                
            if args.format == "json":
                alerts_data = []
                for alert in alerts:
                    alerts_data.append({
                        "id": alert.id,
                        "rule_id": alert.rule_id,
                        "rule_name": alert.rule_name,
                        "severity": alert.severity.value,
                        "status": alert.status.value,
                        "message": alert.message,
                        "metric_value": alert.metric_value,
                        "threshold": alert.threshold,
                        "started_at": alert.started_at.isoformat() if alert.started_at else None,
                        "acknowledged_by": alert.acknowledged_by,
                        "suppressed": alert.suppressed,
                        "labels": alert.labels,
                        "annotations": alert.annotations
                    })
                print(json.dumps(alerts_data, indent=2, ensure_ascii=False))
            else:
                print(f"{'ID':<20} {'规则':<20} {'严重级别':<10} {'状态':<10} {'开始时间':<20}")
                print("-" * 90)
                for alert in alerts:
                    started = alert.started_at.strftime("%Y-%m-%d %H:%M:%S") if alert.started_at else "N/A"
                    print(f"{alert.id:<20} {alert.rule_id:<20} {alert.severity.value:<10} {alert.status.value:<10} {started:<20}")
                    
        finally:
            await self.cleanup()
            
        return 0
        
    async def cmd_acknowledge_alert(self, args):
        """确认告警"""
        await self.initialize(args.db)
        
        try:
            await self.alert_manager.acknowledge_alert(args.id, args.user)
            print(f"成功确认告警: {args.id}")
            
        except Exception as e:
            print(f"确认告警失败: {e}")
            return 1
        finally:
            await self.cleanup()
            
        return 0
        
    # 抑制管理命令
    async def cmd_list_suppressions(self, args):
        """列出抑制规则"""
        await self.initialize(args.db)
        
        try:
            rules = await self.suppression_manager.get_suppression_rules()
            
            if args.format == "json":
                rules_data = []
                for rule in rules:
                    rules_data.append({
                        "id": rule.id,
                        "name": rule.name,
                        "type": rule.type.value,
                        "target_labels": rule.target_labels,
                        "start_time": rule.start_time.isoformat() if rule.start_time else None,
                        "end_time": rule.end_time.isoformat() if rule.end_time else None,
                        "reason": rule.reason,
                        "enabled": rule.enabled
                    })
                print(json.dumps(rules_data, indent=2, ensure_ascii=False))
            else:
                print(f"{'ID':<20} {'名称':<30} {'类型':<15} {'状态':<10}")
                print("-" * 85)
                for rule in rules:
                    status = "启用" if rule.enabled else "禁用"
                    print(f"{rule.id:<20} {rule.name:<30} {rule.type.value:<15} {status:<10}")
                    
        finally:
            await self.cleanup()
            
        return 0
        
    async def cmd_add_suppression(self, args):
        """添加抑制规则"""
        await self.initialize(args.db)
        
        try:
            # 解析时间
            start_time = datetime.fromisoformat(args.start_time) if args.start_time else datetime.now()
            end_time = datetime.fromisoformat(args.end_time) if args.end_time else start_time + timedelta(hours=1)
            
            # 解析标签
            target_labels = json.loads(args.labels) if args.labels else {}
            
            rule = SuppressionRule(
                id=args.id,
                name=args.name,
                type=SuppressionType(args.type),
                target_labels=target_labels,
                start_time=start_time,
                end_time=end_time,
                reason=args.reason or "手动添加"
            )
            
            await self.suppression_manager.add_suppression_rule(rule)
            print(f"成功添加抑制规则: {rule.id}")
            
        except Exception as e:
            print(f"添加抑制规则失败: {e}")
            return 1
        finally:
            await self.cleanup()
            
        return 0
        
    # 历史查询命令
    async def cmd_history(self, args):
        """查询告警历史"""
        await self.initialize(args.db)
        
        try:
            # 解析时间范围
            start_time = datetime.fromisoformat(args.start_time) if args.start_time else None
            end_time = datetime.fromisoformat(args.end_time) if args.end_time else None
            
            records = await self.alert_history.get_alert_history(
                rule_id=args.rule_id,
                severity=args.severity,
                start_time=start_time,
                end_time=end_time,
                limit=args.limit,
                offset=args.offset
            )
            
            if args.format == "json":
                records_data = []
                for record in records:
                    records_data.append({
                        "alert_id": record.alert_id,
                        "rule_id": record.rule_id,
                        "rule_name": record.rule_name,
                        "severity": record.severity,
                        "status": record.status,
                        "message": record.message,
                        "metric_value": record.metric_value,
                        "threshold": record.threshold,
                        "started_at": record.started_at.isoformat() if record.started_at else None,
                        "resolved_at": record.resolved_at.isoformat() if record.resolved_at else None,
                        "acknowledged_by": record.acknowledged_by,
                        "labels": record.labels,
                        "annotations": record.annotations
                    })
                print(json.dumps(records_data, indent=2, ensure_ascii=False))
            else:
                print(f"{'告警ID':<20} {'规则ID':<20} {'严重级别':<10} {'状态':<10} {'开始时间':<20}")
                print("-" * 90)
                for record in records:
                    started = record.started_at.strftime("%Y-%m-%d %H:%M:%S") if record.started_at else "N/A"
                    print(f"{record.alert_id:<20} {record.rule_id:<20} {record.severity:<10} {record.status:<10} {started:<20}")
                    
        finally:
            await self.cleanup()
            
        return 0
        
    async def cmd_statistics(self, args):
        """查询统计信息"""
        await self.initialize(args.db)
        
        try:
            stats = await self.alert_history.get_statistics(
                start_time=datetime.fromisoformat(args.start_time) if args.start_time else None,
                end_time=datetime.fromisoformat(args.end_time) if args.end_time else None
            )
            
            if args.format == "json":
                print(json.dumps(stats, indent=2, ensure_ascii=False))
            else:
                print("=== 告警统计信息 ===")
                print(f"总告警数: {stats.get('total_alerts', 0)}")
                print(f"活跃告警数: {stats.get('active_alerts', 0)}")
                print(f"已解决告警数: {stats.get('resolved_alerts', 0)}")
                print(f"已确认告警数: {stats.get('acknowledged_alerts', 0)}")
                
                if 'severity_distribution' in stats:
                    print("\n严重级别分布:")
                    for severity, count in stats['severity_distribution'].items():
                        print(f"  {severity}: {count}")
                        
                if 'top_rules' in stats:
                    print("\n最活跃规则:")
                    for rule in stats['top_rules']:
                        print(f"  {rule['rule_id']}: {rule['count']} 次")
                        
        finally:
            await self.cleanup()
            
        return 0


def create_parser():
    """创建命令行解析器"""
    parser = argparse.ArgumentParser(
        description="HarborAI 告警系统命令行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # 全局选项
    parser.add_argument("--db", help="数据库文件路径")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="输出格式")
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 配置管理命令
    config_parser = subparsers.add_parser("config", help="配置管理")
    config_subparsers = config_parser.add_subparsers(dest="config_command")
    
    # 验证配置
    validate_parser = config_subparsers.add_parser("validate", help="验证配置文件")
    validate_parser.add_argument("config", help="配置文件路径")
    
    # 生成配置
    generate_parser = config_subparsers.add_parser("generate", help="生成配置文件")
    generate_parser.add_argument("--type", choices=["default", "production", "development"], 
                               default="default", help="配置类型")
    generate_parser.add_argument("--output", help="输出文件路径")
    
    # 规则管理命令
    rules_parser = subparsers.add_parser("rules", help="规则管理")
    rules_subparsers = rules_parser.add_subparsers(dest="rules_command")
    
    # 列出规则
    list_rules_parser = rules_subparsers.add_parser("list", help="列出告警规则")
    
    # 添加规则
    add_rule_parser = rules_subparsers.add_parser("add", help="添加告警规则")
    add_rule_parser.add_argument("--file", help="从文件加载规则")
    add_rule_parser.add_argument("--id", help="规则ID")
    add_rule_parser.add_argument("--name", help="规则名称")
    add_rule_parser.add_argument("--description", help="规则描述")
    add_rule_parser.add_argument("--severity", choices=["low", "medium", "high", "critical"], help="严重级别")
    add_rule_parser.add_argument("--condition", choices=["threshold", "anomaly"], help="条件类型")
    add_rule_parser.add_argument("--metric", help="指标名称")
    add_rule_parser.add_argument("--threshold", type=float, help="阈值")
    add_rule_parser.add_argument("--duration", type=int, default=60, help="持续时间（秒）")
    add_rule_parser.add_argument("--labels", help="标签（JSON格式）")
    add_rule_parser.add_argument("--annotations", help="注解（JSON格式）")
    
    # 删除规则
    remove_rule_parser = rules_subparsers.add_parser("remove", help="删除告警规则")
    remove_rule_parser.add_argument("id", help="规则ID")
    
    # 告警管理命令
    alerts_parser = subparsers.add_parser("alerts", help="告警管理")
    alerts_subparsers = alerts_parser.add_subparsers(dest="alerts_command")
    
    # 列出告警
    list_alerts_parser = alerts_subparsers.add_parser("list", help="列出活跃告警")
    list_alerts_parser.add_argument("--severity", choices=["low", "medium", "high", "critical"], help="按严重级别过滤")
    
    # 确认告警
    ack_parser = alerts_subparsers.add_parser("acknowledge", help="确认告警")
    ack_parser.add_argument("id", help="告警ID")
    ack_parser.add_argument("--user", default="cli", help="确认用户")
    
    # 抑制管理命令
    suppression_parser = subparsers.add_parser("suppression", help="抑制管理")
    suppression_subparsers = suppression_parser.add_subparsers(dest="suppression_command")
    
    # 列出抑制规则
    list_suppression_parser = suppression_subparsers.add_parser("list", help="列出抑制规则")
    
    # 添加抑制规则
    add_suppression_parser = suppression_subparsers.add_parser("add", help="添加抑制规则")
    add_suppression_parser.add_argument("id", help="抑制规则ID")
    add_suppression_parser.add_argument("name", help="抑制规则名称")
    add_suppression_parser.add_argument("--type", choices=["time_based", "label_based", "pattern_based"], 
                                      default="time_based", help="抑制类型")
    add_suppression_parser.add_argument("--labels", help="目标标签（JSON格式）")
    add_suppression_parser.add_argument("--start-time", help="开始时间（ISO格式）")
    add_suppression_parser.add_argument("--end-time", help="结束时间（ISO格式）")
    add_suppression_parser.add_argument("--reason", help="抑制原因")
    
    # 历史查询命令
    history_parser = subparsers.add_parser("history", help="查询告警历史")
    history_parser.add_argument("--rule-id", help="按规则ID过滤")
    history_parser.add_argument("--severity", help="按严重级别过滤")
    history_parser.add_argument("--start-time", help="开始时间（ISO格式）")
    history_parser.add_argument("--end-time", help="结束时间（ISO格式）")
    history_parser.add_argument("--limit", type=int, default=50, help="返回记录数限制")
    history_parser.add_argument("--offset", type=int, default=0, help="记录偏移量")
    
    # 统计信息命令
    stats_parser = subparsers.add_parser("stats", help="查询统计信息")
    stats_parser.add_argument("--start-time", help="开始时间（ISO格式）")
    stats_parser.add_argument("--end-time", help="结束时间（ISO格式）")
    
    return parser


async def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
        
    cli = AlertCLI()
    
    try:
        # 配置管理命令
        if args.command == "config":
            if args.config_command == "validate":
                return await cli.cmd_validate_config(args)
            elif args.config_command == "generate":
                return await cli.cmd_generate_config(args)
                
        # 规则管理命令
        elif args.command == "rules":
            if args.rules_command == "list":
                return await cli.cmd_list_rules(args)
            elif args.rules_command == "add":
                return await cli.cmd_add_rule(args)
            elif args.rules_command == "remove":
                return await cli.cmd_remove_rule(args)
                
        # 告警管理命令
        elif args.command == "alerts":
            if args.alerts_command == "list":
                return await cli.cmd_list_alerts(args)
            elif args.alerts_command == "acknowledge":
                return await cli.cmd_acknowledge_alert(args)
                
        # 抑制管理命令
        elif args.command == "suppression":
            if args.suppression_command == "list":
                return await cli.cmd_list_suppressions(args)
            elif args.suppression_command == "add":
                return await cli.cmd_add_suppression(args)
                
        # 历史查询命令
        elif args.command == "history":
            return await cli.cmd_history(args)
            
        # 统计信息命令
        elif args.command == "stats":
            return await cli.cmd_statistics(args)
            
        else:
            print(f"未知命令: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        print("\n操作被用户中断")
        return 1
    except Exception as e:
        print(f"执行命令时发生错误: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))