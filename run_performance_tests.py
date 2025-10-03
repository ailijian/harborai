#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI SDK 性能测试执行脚本

一键执行完整的性能测试流程：
1. 环境检查和准备
2. 启动HarborAI服务
3. 执行综合性能测试
4. 生成性能报告
5. 清理测试环境
"""

import asyncio
import json
import os
import sys
import time
import subprocess
import signal
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceTestRunner:
    """性能测试运行器
    
    负责协调整个性能测试流程的执行
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.test_dir = self.project_root / "tests" / "performance"
        self.server_process = None
        self.test_results_file = None
        
        # 测试配置
        self.config = {
            "server_host": "localhost",
            "server_port": 8000,
            "server_startup_timeout": 30,  # 30秒
            "test_timeout": 1800,  # 30分钟
            "cleanup_timeout": 10   # 10秒
        }
    
    async def run_complete_test_suite(self) -> bool:
        """运行完整的测试套件"""
        logger.info("🚀 开始执行HarborAI SDK完整性能测试")
        
        try:
            # 1. 环境检查
            logger.info("1️⃣ 检查测试环境...")
            if not await self._check_environment():
                logger.error("❌ 环境检查失败")
                return False
            
            # 2. 启动HarborAI服务
            logger.info("2️⃣ 启动HarborAI服务...")
            if not await self._start_harborai_server():
                logger.error("❌ 服务启动失败")
                return False
            
            # 3. 等待服务就绪
            logger.info("3️⃣ 等待服务就绪...")
            if not await self._wait_for_server_ready():
                logger.error("❌ 服务未能正常启动")
                return False
            
            # 4. 执行性能测试
            logger.info("4️⃣ 执行综合性能测试...")
            if not await self._run_performance_tests():
                logger.error("❌ 性能测试执行失败")
                return False
            
            # 5. 生成性能报告
            logger.info("5️⃣ 生成性能报告...")
            if not await self._generate_performance_report():
                logger.error("❌ 报告生成失败")
                return False
            
            logger.info("✅ 性能测试完成！")
            return True
            
        except Exception as e:
            logger.error(f"❌ 测试执行失败: {e}")
            return False
        
        finally:
            # 6. 清理测试环境
            logger.info("6️⃣ 清理测试环境...")
            await self._cleanup()
    
    async def _check_environment(self) -> bool:
        """检查测试环境"""
        try:
            # 检查Python版本
            python_version = sys.version_info
            if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
                logger.error("需要Python 3.8或更高版本")
                return False
            
            logger.info(f"✅ Python版本: {python_version.major}.{python_version.minor}")
            
            # 检查必要的依赖
            required_packages = [
                "pytest", "pytest-asyncio", "pytest-benchmark",
                "httpx", "psutil", "aiohttp"
            ]
            
            missing_packages = []
            for package in required_packages:
                try:
                    __import__(package.replace("-", "_"))
                except ImportError:
                    missing_packages.append(package)
            
            if missing_packages:
                logger.error(f"缺少必要依赖: {', '.join(missing_packages)}")
                logger.info("请运行: pip install -r requirements-test.txt")
                return False
            
            logger.info("✅ 依赖检查通过")
            
            # 检查测试文件
            test_files = [
                "tests/performance/comprehensive_performance_test.py",
                "tests/performance/performance_report_generator.py"
            ]
            
            for test_file in test_files:
                if not (self.project_root / test_file).exists():
                    logger.error(f"测试文件不存在: {test_file}")
                    return False
            
            logger.info("✅ 测试文件检查通过")
            
            # 创建报告目录
            report_dir = self.test_dir / "performance_reports"
            report_dir.mkdir(exist_ok=True)
            
            return True
            
        except Exception as e:
            logger.error(f"环境检查失败: {e}")
            return False
    
    async def _start_harborai_server(self) -> bool:
        """启动HarborAI服务"""
        try:
            # 检查是否已有服务在运行
            if await self._is_server_running():
                logger.info("检测到HarborAI服务已在运行")
                return True
            
            # 查找启动脚本
            possible_start_scripts = [
                "start_server.py",
                "main.py",
                "app.py",
                "server.py"
            ]
            
            start_script = None
            for script in possible_start_scripts:
                script_path = self.project_root / script
                if script_path.exists():
                    start_script = script_path
                    break
            
            if not start_script:
                # 尝试使用uvicorn启动
                logger.info("未找到启动脚本，尝试使用uvicorn启动...")
                cmd = [
                    sys.executable, "-m", "uvicorn",
                    "harborai.api.main:app",
                    "--host", self.config["server_host"],
                    "--port", str(self.config["server_port"]),
                    "--reload"
                ]
            else:
                logger.info(f"使用启动脚本: {start_script}")
                cmd = [sys.executable, str(start_script)]
            
            # 启动服务
            logger.info(f"执行命令: {' '.join(cmd)}")
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.project_root)
            )
            
            logger.info(f"HarborAI服务已启动 (PID: {self.server_process.pid})")
            return True
            
        except Exception as e:
            logger.error(f"启动服务失败: {e}")
            return False
    
    async def _wait_for_server_ready(self) -> bool:
        """等待服务就绪"""
        import httpx
        
        base_url = f"http://{self.config['server_host']}:{self.config['server_port']}"
        timeout = self.config["server_startup_timeout"]
        
        logger.info(f"等待服务就绪: {base_url}")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                async with httpx.AsyncClient() as client:
                    # 尝试访问健康检查端点
                    response = await client.get(f"{base_url}/health", timeout=5.0)
                    if response.status_code == 200:
                        logger.info("✅ 服务就绪")
                        return True
            except:
                pass
            
            # 检查服务进程是否还在运行
            if self.server_process and self.server_process.poll() is not None:
                logger.error("服务进程已退出")
                return False
            
            await asyncio.sleep(2)
        
        logger.error(f"服务在{timeout}秒内未就绪")
        return False
    
    async def _is_server_running(self) -> bool:
        """检查服务是否在运行"""
        import httpx
        
        try:
            base_url = f"http://{self.config['server_host']}:{self.config['server_port']}"
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{base_url}/health", timeout=5.0)
                return response.status_code == 200
        except:
            return False
    
    async def _run_performance_tests(self) -> bool:
        """运行性能测试"""
        try:
            # 导入测试模块
            sys.path.insert(0, str(self.test_dir))
            from comprehensive_performance_test import ComprehensivePerformanceTester
            
            # 创建测试器
            config = {
                "base_url": f"http://{self.config['server_host']}:{self.config['server_port']}",
                "api_key": os.getenv("HARBORAI_API_KEY", "test-key")
            }
            
            tester = ComprehensivePerformanceTester(config)
            
            # 初始化测试器
            if not await tester.initialize():
                logger.error("测试器初始化失败")
                return False
            
            # 运行测试
            logger.info("开始执行性能测试...")
            results = await asyncio.wait_for(
                tester.run_comprehensive_tests(),
                timeout=self.config["test_timeout"]
            )
            
            # 保存结果
            self.test_results_file = await tester.save_results()
            
            # 清理测试器
            await tester.cleanup()
            
            logger.info("✅ 性能测试完成")
            return True
            
        except asyncio.TimeoutError:
            logger.error(f"性能测试超时 ({self.config['test_timeout']}秒)")
            return False
        except Exception as e:
            logger.error(f"性能测试失败: {e}")
            return False
    
    async def _generate_performance_report(self) -> bool:
        """生成性能报告"""
        try:
            if not self.test_results_file:
                logger.error("没有测试结果文件")
                return False
            
            # 导入报告生成器
            sys.path.insert(0, str(self.test_dir))
            from performance_report_generator import PerformanceReportGenerator
            
            # 生成报告
            generator = PerformanceReportGenerator(self.test_results_file)
            report_path = generator.generate_comprehensive_report()
            
            logger.info(f"✅ 性能报告已生成: {report_path}")
            
            # 打印报告摘要
            await self._print_report_summary()
            
            return True
            
        except Exception as e:
            logger.error(f"报告生成失败: {e}")
            return False
    
    async def _print_report_summary(self):
        """打印报告摘要"""
        try:
            if not self.test_results_file or not Path(self.test_results_file).exists():
                return
            
            with open(self.test_results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            print("\n" + "="*80)
            print("🎯 HarborAI SDK 性能测试摘要")
            print("="*80)
            
            # 分析结果
            analysis = results.get("analysis", {})
            prd_compliance = analysis.get("prd_compliance", {})
            bottlenecks = analysis.get("bottlenecks", [])
            recommendations = analysis.get("recommendations", [])
            
            # PRD合规性
            if prd_compliance:
                print("\n📊 PRD合规性检查:")
                for metric, data in prd_compliance.items():
                    status = "✅" if data.get("compliant") else "❌"
                    print(f"  {status} {metric}: {data.get('actual')} (要求: {data.get('requirement')})")
                
                compliant_count = sum(1 for data in prd_compliance.values() if data.get("compliant"))
                total_count = len(prd_compliance)
                compliance_rate = (compliant_count / total_count * 100) if total_count > 0 else 0
                print(f"\n  📈 总体合规率: {compliance_rate:.1f}% ({compliant_count}/{total_count})")
            
            # 性能瓶颈
            if bottlenecks:
                print(f"\n⚠️ 发现性能瓶颈 ({len(bottlenecks)} 项):")
                for bottleneck in bottlenecks[:3]:  # 只显示前3个
                    print(f"  • {bottleneck.get('description')}: {bottleneck.get('value')}")
                if len(bottlenecks) > 3:
                    print(f"  • ... 还有 {len(bottlenecks) - 3} 项")
            else:
                print("\n✅ 未发现明显性能瓶颈")
            
            # 优化建议
            if recommendations:
                high_priority = [r for r in recommendations if r.get("priority") == "high"]
                if high_priority:
                    print(f"\n🔧 高优先级优化建议 ({len(high_priority)} 项):")
                    for rec in high_priority[:3]:
                        print(f"  • {rec.get('description')}")
            
            # 测试统计
            performance_summary = analysis.get("performance_summary", {})
            if performance_summary:
                duration = performance_summary.get("total_test_duration", 0)
                print(f"\n⏱️ 测试执行时间: {duration:.1f} 秒")
            
            print("\n" + "="*80)
            print(f"📄 详细报告文件: {self.test_results_file}")
            print("="*80)
            
        except Exception as e:
            logger.error(f"打印摘要失败: {e}")
    
    async def _cleanup(self):
        """清理测试环境"""
        try:
            # 停止服务进程
            if self.server_process:
                logger.info("停止HarborAI服务...")
                try:
                    # 尝试优雅关闭
                    self.server_process.terminate()
                    
                    # 等待进程结束
                    try:
                        self.server_process.wait(timeout=self.config["cleanup_timeout"])
                    except subprocess.TimeoutExpired:
                        # 强制杀死进程
                        logger.warning("强制终止服务进程")
                        self.server_process.kill()
                        self.server_process.wait()
                    
                    logger.info("✅ 服务已停止")
                    
                except Exception as e:
                    logger.error(f"停止服务失败: {e}")
            
            # 清理临时文件（如果有的话）
            # 这里可以添加清理逻辑
            
            logger.info("✅ 环境清理完成")
            
        except Exception as e:
            logger.error(f"清理失败: {e}")


def signal_handler(signum, frame):
    """信号处理器"""
    logger.info("收到中断信号，正在清理...")
    sys.exit(0)


async def main():
    """主函数"""
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 创建测试运行器
    runner = PerformanceTestRunner()
    
    try:
        # 运行完整测试套件
        success = await runner.run_complete_test_suite()
        
        if success:
            print("\n🎉 HarborAI SDK性能测试成功完成！")
            sys.exit(0)
        else:
            print("\n❌ HarborAI SDK性能测试失败！")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("用户中断测试")
        await runner._cleanup()
        sys.exit(1)
    except Exception as e:
        logger.error(f"测试运行失败: {e}")
        await runner._cleanup()
        sys.exit(1)


if __name__ == "__main__":
    # 设置事件循环策略（Windows兼容性）
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())