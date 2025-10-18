#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
告警系统启动脚本

提供多种启动方式：
- 后台服务模式
- Web界面模式
- 命令行工具模式
"""

import asyncio
import argparse
import signal
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from harborai.core.alerts import (
    start_alert_system,
    start_web_ui,
    run_cli,
    get_default_config,
    get_production_config,
    get_development_config
)


class AlertSystemLauncher:
    """告警系统启动器"""
    
    def __init__(self):
        self.running = False
        self.components = None
        
    async def start_service(self, args):
        """启动后台服务模式"""
        print("启动告警系统后台服务...")
        
        try:
            self.components = await start_alert_system(
                db_path=args.db_path,
                config_path=args.config,
                enable_web_ui=args.web_ui,
                web_port=args.port
            )
            
            self.running = True
            print("告警系统启动成功")
            
            # 设置信号处理
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            # 保持运行
            while self.running:
                await asyncio.sleep(1)
                
        except Exception as e:
            print(f"启动告警系统失败: {e}")
            return 1
            
        return 0
        
    async def start_web_only(self, args):
        """仅启动Web界面"""
        print("启动告警系统Web界面...")
        
        try:
            await start_web_ui(
                db_path=args.db_path,
                port=args.port
            )
            
        except Exception as e:
            print(f"启动Web界面失败: {e}")
            return 1
            
        return 0
        
    def start_cli(self, args):
        """启动命令行工具"""
        # 设置命令行参数
        cli_args = []
        
        if args.command:
            cli_args.extend(args.command)
            
        if args.db_path:
            cli_args.extend(["--db", args.db_path])
            
        if args.format:
            cli_args.extend(["--format", args.format])
            
        # 临时修改sys.argv
        original_argv = sys.argv[:]
        sys.argv = ["alert-cli"] + cli_args
        
        try:
            run_cli()
        finally:
            sys.argv = original_argv
            
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        print(f"\n接收到信号 {signum}，正在关闭告警系统...")
        self.running = False
        
    async def stop_service(self):
        """停止服务"""
        if self.components:
            alert_manager, _, _, _ = self.components
            await alert_manager.stop()
            print("告警系统已停止")


def create_parser():
    """创建命令行解析器"""
    parser = argparse.ArgumentParser(
        description="HarborAI 告警系统启动器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 启动完整的告警系统（包含Web界面）
  python start_alert_system.py service --web-ui --port 8080
  
  # 仅启动Web界面
  python start_alert_system.py web --port 8080
  
  # 使用命令行工具
  python start_alert_system.py cli rules list
  python start_alert_system.py cli alerts list --severity critical
  
  # 生成配置文件
  python start_alert_system.py cli config generate --type production --output alerts.json
        """
    )
    
    # 全局选项
    parser.add_argument("--db-path", default="alerts.db", help="数据库文件路径")
    parser.add_argument("--config", help="配置文件路径")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    
    subparsers = parser.add_subparsers(dest="mode", help="运行模式")
    
    # 服务模式
    service_parser = subparsers.add_parser("service", help="启动后台服务")
    service_parser.add_argument("--web-ui", action="store_true", help="启用Web界面")
    service_parser.add_argument("--port", type=int, default=8080, help="Web界面端口")
    
    # Web界面模式
    web_parser = subparsers.add_parser("web", help="仅启动Web界面")
    web_parser.add_argument("--port", type=int, default=8080, help="Web界面端口")
    
    # 命令行模式
    cli_parser = subparsers.add_parser("cli", help="命令行工具")
    cli_parser.add_argument("--format", choices=["text", "json"], default="text", help="输出格式")
    cli_parser.add_argument("command", nargs="*", help="CLI命令和参数")
    
    # 配置生成模式
    config_parser = subparsers.add_parser("generate-config", help="生成配置文件")
    config_parser.add_argument("--type", choices=["default", "production", "development"], 
                              default="default", help="配置类型")
    config_parser.add_argument("--output", help="输出文件路径")
    
    return parser


async def generate_config(args):
    """生成配置文件"""
    import json
    
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


async def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        return 1
        
    # 设置日志级别
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
        
    launcher = AlertSystemLauncher()
    
    try:
        if args.mode == "service":
            return await launcher.start_service(args)
        elif args.mode == "web":
            return await launcher.start_web_only(args)
        elif args.mode == "cli":
            return launcher.start_cli(args)
        elif args.mode == "generate-config":
            return await generate_config(args)
        else:
            print(f"未知模式: {args.mode}")
            return 1
            
    except KeyboardInterrupt:
        print("\n操作被用户中断")
        await launcher.stop_service()
        return 1
    except Exception as e:
        print(f"执行失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    # 确保在Windows上正确处理异步
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
    sys.exit(asyncio.run(main()))