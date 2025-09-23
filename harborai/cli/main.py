#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI CLI 主命令

提供数据库初始化、插件管理、日志查看等命令行功能。
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Optional, List

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax

from ..database.connection import init_database_sync, get_db_session
from ..database.models import APILog, TraceLog, ModelUsage
from ..config.settings import get_settings
from ..core.client_manager import ClientManager
from ..utils.logger import get_logger

console = Console()
logger = get_logger("harborai.cli")


@click.group()
@click.version_option()
def cli():
    """HarborAI - 统一的 LLM 调用接口"""
    pass


@cli.command()
@click.option(
    "--force",
    is_flag=True,
    help="强制重新创建数据库表"
)
def init_db(force: bool):
    """初始化数据库"""
    console.print("[bold blue]初始化 HarborAI 数据库...[/bold blue]")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("创建数据库表...", total=None)
            
            init_database_sync()
            
            progress.update(task, description="数据库初始化完成")
        
        console.print("[bold green]✓ 数据库初始化成功![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]✗ 数据库初始化失败: {e}[/bold red]")
        raise click.ClickException(str(e))


@cli.command()
def list_plugins():
    """列出所有可用插件"""
    console.print("[bold blue]HarborAI 插件列表[/bold blue]")
    
    try:
        client_manager = ClientManager()
        plugin_info = client_manager.get_plugin_info()
        
        if not plugin_info:
            console.print("[yellow]未找到任何插件[/yellow]")
            return
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("插件名称", style="cyan")
        table.add_column("支持模型数", justify="center")
        table.add_column("支持的模型", style="green")
        
        for plugin_name, info in plugin_info.items():
            models = ", ".join(info["supported_models"][:3])  # 只显示前3个
            if len(info["supported_models"]) > 3:
                models += f" (+{len(info['supported_models']) - 3} 更多)"
            
            table.add_row(
                plugin_name,
                str(info["model_count"]),
                models
            )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[bold red]✗ 获取插件信息失败: {e}[/bold red]")
        raise click.ClickException(str(e))


@cli.command()
def list_models():
    """列出所有可用模型"""
    console.print("[bold blue]HarborAI 模型列表[/bold blue]")
    
    try:
        client_manager = ClientManager()
        models = client_manager.get_available_models()
        
        if not models:
            console.print("[yellow]未找到任何模型[/yellow]")
            return
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("模型名称", style="cyan")
        table.add_column("提供商", style="blue")
        table.add_column("思考模型", justify="center")
        table.add_column("结构化输出", justify="center")
        table.add_column("最大Token", justify="right")
        table.add_column("上下文窗口", justify="right")
        
        for model in models:
            table.add_row(
                model.name,
                model.provider,
                "✓" if model.supports_thinking else "✗",
                "✓" if model.supports_structured_output else "✗",
                f"{model.max_tokens:,}" if model.max_tokens else "N/A",
                f"{model.context_window:,}" if model.context_window else "N/A"
            )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[bold red]✗ 获取模型信息失败: {e}[/bold red]")
        raise click.ClickException(str(e))


@cli.command()
@click.option(
    "--days",
    default=7,
    help="查看最近几天的日志 (默认: 7)"
)
@click.option(
    "--model",
    help="过滤特定模型的日志"
)
@click.option(
    "--plugin",
    help="过滤特定插件的日志"
)
@click.option(
    "--limit",
    default=50,
    help="限制显示的日志条数 (默认: 50)"
)
def logs(days: int, model: Optional[str], plugin: Optional[str], limit: int):
    """查看 API 调用日志"""
    console.print(f"[bold blue]HarborAI API 日志 (最近 {days} 天)[/bold blue]")
    
    try:
        with get_db_session() as session:
            if session is None:
                console.print("[yellow]数据库未启用，无法查看日志[/yellow]")
                return
            
            # 构建查询
            query = session.query(APILog)
            
            # 时间过滤
            since_date = datetime.utcnow() - timedelta(days=days)
            query = query.filter(APILog.timestamp >= since_date)
            
            # 模型过滤
            if model:
                query = query.filter(APILog.model == model)
            
            # 插件过滤
            if plugin:
                query = query.filter(APILog.plugin == plugin)
            
            # 排序和限制
            logs = query.order_by(APILog.timestamp.desc()).limit(limit).all()
            
            if not logs:
                console.print("[yellow]未找到匹配的日志记录[/yellow]")
                return
            
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("时间", style="cyan")
            table.add_column("模型", style="blue")
            table.add_column("插件", style="green")
            table.add_column("状态", justify="center")
            table.add_column("Token", justify="right")
            table.add_column("耗时(ms)", justify="right")
            table.add_column("成本", justify="right")
            
            for log in logs:
                status_style = "green" if log.response_status == "success" else "red"
                status = f"[{status_style}]{log.response_status or 'unknown'}[/{status_style}]"
                
                tokens = f"{log.total_tokens:,}" if log.total_tokens else "N/A"
                duration = f"{log.duration_ms:.1f}" if log.duration_ms else "N/A"
                cost = f"${log.estimated_cost:.4f}" if log.estimated_cost else "N/A"
                
                table.add_row(
                    log.timestamp.strftime("%m-%d %H:%M:%S") if log.timestamp else "N/A",
                    log.model or "N/A",
                    log.plugin or "N/A",
                    status,
                    tokens,
                    duration,
                    cost
                )
            
            console.print(table)
            
    except Exception as e:
        console.print(f"[bold red]✗ 查看日志失败: {e}[/bold red]")
        raise click.ClickException(str(e))


@cli.command()
@click.option(
    "--days",
    default=30,
    help="统计最近几天的使用情况 (默认: 30)"
)
def stats(days: int):
    """查看使用统计"""
    console.print(f"[bold blue]HarborAI 使用统计 (最近 {days} 天)[/bold blue]")
    
    try:
        with get_db_session() as session:
            if session is None:
                console.print("[yellow]数据库未启用，无法查看统计[/yellow]")
                return
            
            # 时间过滤
            since_date = datetime.utcnow() - timedelta(days=days)
            
            # 总体统计
            total_requests = session.query(APILog).filter(
                APILog.timestamp >= since_date
            ).count()
            
            success_requests = session.query(APILog).filter(
                APILog.timestamp >= since_date,
                APILog.response_status == "success"
            ).count()
            
            # 按模型统计
            from sqlalchemy import func
            model_stats = session.query(
                APILog.model,
                func.count(APILog.id).label('request_count'),
                func.sum(APILog.total_tokens).label('total_tokens'),
                func.sum(APILog.estimated_cost).label('total_cost')
            ).filter(
                APILog.timestamp >= since_date,
                APILog.model.isnot(None)
            ).group_by(APILog.model).all()
            
            # 显示总体统计
            success_rate = (success_requests / total_requests * 100) if total_requests > 0 else 0
            
            summary_panel = Panel(
                f"总请求数: {total_requests:,}\n"
                f"成功请求: {success_requests:,}\n"
                f"成功率: {success_rate:.1f}%",
                title="总体统计",
                border_style="blue"
            )
            console.print(summary_panel)
            
            # 显示模型统计
            if model_stats:
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("模型", style="cyan")
                table.add_column("请求数", justify="right")
                table.add_column("总Token", justify="right")
                table.add_column("总成本", justify="right")
                
                for stat in model_stats:
                    tokens = f"{stat.total_tokens:,}" if stat.total_tokens else "0"
                    cost = f"${stat.total_cost:.4f}" if stat.total_cost else "$0.0000"
                    
                    table.add_row(
                        stat.model,
                        f"{stat.request_count:,}",
                        tokens,
                        cost
                    )
                
                console.print("\n[bold blue]按模型统计[/bold blue]")
                console.print(table)
            
    except Exception as e:
        console.print(f"[bold red]✗ 查看统计失败: {e}[/bold red]")
        raise click.ClickException(str(e))


@cli.command()
def config():
    """显示当前配置"""
    console.print("[bold blue]HarborAI 配置信息[/bold blue]")
    
    try:
        settings = get_settings()
        
        # 基础配置
        basic_config = {
            "默认超时": f"{settings.default_timeout}s",
            "最大重试次数": settings.max_retries,
            "数据库启用": "是" if settings.enable_database else "否",
            "日志级别": settings.log_level,
            "结构化输出提供商": settings.structured_output_provider
        }
        
        basic_panel = Panel(
            "\n".join([f"{k}: {v}" for k, v in basic_config.items()]),
            title="基础配置",
            border_style="green"
        )
        console.print(basic_panel)
        
        # 插件目录
        if settings.plugin_directories:
            plugin_panel = Panel(
                "\n".join(settings.plugin_directories),
                title="插件目录",
                border_style="blue"
            )
            console.print(plugin_panel)
        
        # 数据库配置（如果启用）
        if settings.enable_database:
            db_config = {
                "主机": settings.postgres_host,
                "端口": settings.postgres_port,
                "数据库": settings.postgres_db,
                "用户": settings.postgres_user,
                "连接池大小": settings.db_pool_size,
                "最大溢出": settings.db_max_overflow
            }
            
            db_panel = Panel(
                "\n".join([f"{k}: {v}" for k, v in db_config.items()]),
                title="数据库配置",
                border_style="yellow"
            )
            console.print(db_panel)
        
    except Exception as e:
        console.print(f"[bold red]✗ 获取配置失败: {e}[/bold red]")
        raise click.ClickException(str(e))


if __name__ == "__main__":
    cli()