"""
HarborAI性能测试主运行器

该脚本提供完整的性能测试执行入口，支持：
- 单独运行各个性能测试模块
- 运行完整的集成测试套件
- 生成详细的性能报告
- 命令行参数配置
- 测试结果持久化

使用方法:
    python run_performance_tests.py --help
    python run_performance_tests.py --all
    python run_performance_tests.py --module response_time
    python run_performance_tests.py --integration --output ./reports

作者：HarborAI性能测试团队
创建时间：2024年
"""

import argparse
import asyncio
import sys
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import traceback

# 添加项目根目录到 Python 路径
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 导入性能测试模块
from tests.performance.core_performance_framework import (
    PerformanceTestController,
    PerformanceConfig,
    TestType,
    TestStatus
)
from tests.performance.memory_leak_detector import MemoryLeakDetector, detect_memory_leak
from tests.performance.resource_utilization_monitor import ResourceUtilizationMonitor
from tests.performance.execution_efficiency_tests import ExecutionEfficiencyTester
from tests.performance.response_time_tests import ResponseTimeTester, test_api_response_time
from tests.performance.concurrency_tests import ConcurrencyTester, test_high_concurrency
from tests.performance.performance_report_generator import PerformanceReportGenerator, generate_quick_report
from tests.performance.test_integration import IntegratedPerformanceTestSuite, run_quick_integration_test

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('performance_tests.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class PerformanceTestRunner:
    """
    性能测试运行器
    
    提供统一的性能测试执行接口，支持：
    - 模块化测试执行
    - 配置管理
    - 结果收集和报告
    - 错误处理和恢复
    """
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        """
        初始化测试运行器
        
        参数:
            config: 性能测试配置
        """
        self.config = config or PerformanceConfig()
        self.test_results: Dict[str, Any] = {}
        self.start_time = datetime.now()
        
        logger.info("性能测试运行器初始化完成")
    
    async def run_response_time_tests(
        self,
        target_url: str = "https://httpbin.org/delay/0.1",
        num_requests: int = 100
    ) -> Dict[str, Any]:
        """
        运行响应时间测试
        
        参数:
            target_url: 测试目标URL
            num_requests: 请求数量
        
        返回:
            测试结果
        """
        logger.info(f"开始响应时间测试: {target_url}")
        
        try:
            tester = ResponseTimeTester()
            
            # 同步测试
            sync_metrics = tester.test_api_response_time(
                target_url,
                num_requests=num_requests,
                test_name="主运行器_同步响应时间测试"
            )
            
            # 异步测试
            async_metrics = await tester.test_async_api_response_time(
                target_url,
                num_requests=num_requests,
                test_name="主运行器_异步响应时间测试"
            )
            
            results = {
                'module': 'response_time',
                'sync_test': sync_metrics.to_dict(),
                'async_test': async_metrics.to_dict(),
                'summary': {
                    'sync_avg_time': sync_metrics.average_response_time,
                    'async_avg_time': async_metrics.average_response_time,
                    'sync_grade': sync_metrics.performance_grade,
                    'async_grade': async_metrics.performance_grade,
                    'better_method': 'sync' if sync_metrics.average_response_time <= async_metrics.average_response_time else 'async'
                }
            }
            
            logger.info(f"响应时间测试完成: 同步 {sync_metrics.average_response_time:.3f}s, "
                       f"异步 {async_metrics.average_response_time:.3f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"响应时间测试失败: {e}")
            return {'module': 'response_time', 'error': str(e), 'traceback': traceback.format_exc()}
    
    async def run_concurrency_tests(
        self,
        target_url: str = "https://httpbin.org/delay/0.1",
        concurrent_users: int = 50,
        requests_per_user: int = 50
    ) -> Dict[str, Any]:
        """
        运行并发处理能力测试
        
        参数:
            target_url: 测试目标URL
            concurrent_users: 并发用户数
            requests_per_user: 每用户请求数
        
        返回:
            测试结果
        """
        logger.info(f"开始并发处理能力测试: {concurrent_users}用户, {requests_per_user}请求/用户")
        
        try:
            # 线程并发测试
            thread_metrics, thread_validation = test_high_concurrency(
                target_url,
                concurrent_users=concurrent_users,
                requests_per_user=requests_per_user,
                test_name="主运行器_线程并发测试"
            )
            
            # 异步并发测试
            async_metrics, async_validation = await test_async_high_concurrency(
                target_url,
                concurrent_users=concurrent_users,
                requests_per_user=requests_per_user,
                test_name="主运行器_异步并发测试"
            )
            
            results = {
                'module': 'concurrency',
                'thread_test': {
                    'metrics': thread_metrics.to_dict(),
                    'validation': thread_validation
                },
                'async_test': {
                    'metrics': async_metrics.to_dict(),
                    'validation': async_validation
                },
                'summary': {
                    'thread_success_rate': thread_metrics.success_rate,
                    'async_success_rate': async_metrics.success_rate,
                    'thread_throughput': thread_metrics.requests_per_second,
                    'async_throughput': async_metrics.requests_per_second,
                    'thread_requirements_met': thread_validation['requirements_met'],
                    'async_requirements_met': async_validation['requirements_met'],
                    'better_method': 'thread' if thread_metrics.requests_per_second >= async_metrics.requests_per_second else 'async'
                }
            }
            
            logger.info(f"并发测试完成: 线程成功率 {thread_metrics.success_rate:.3%}, "
                       f"异步成功率 {async_metrics.success_rate:.3%}")
            
            return results
            
        except Exception as e:
            logger.error(f"并发处理能力测试失败: {e}")
            return {'module': 'concurrency', 'error': str(e), 'traceback': traceback.format_exc()}
    
    async def run_memory_leak_tests(
        self,
        test_duration: int = 60,
        check_interval: int = 10
    ) -> Dict[str, Any]:
        """
        运行内存泄漏检测测试
        
        参数:
            test_duration: 测试持续时间（秒）
            check_interval: 检查间隔（秒）
        
        返回:
            测试结果
        """
        logger.info(f"开始内存泄漏检测测试: {test_duration}秒")
        
        try:
            def memory_intensive_function():
                """内存密集型测试函数"""
                data = []
                for i in range(10000):
                    data.append([j for j in range(100)])
                return len(data)
            
            # 使用便捷函数进行内存泄漏检测
            analysis = detect_memory_leak(
                memory_intensive_function,
                duration=test_duration,
                monitoring_interval=check_interval
            )
            
            results = {
                'module': 'memory_leak',
                'analysis': analysis.to_dict(),
                'summary': {
                    'leak_detected': analysis.is_leak_detected,
                    'leak_rate': analysis.leak_rate,
                    'peak_memory': analysis.peak_memory,
                    'recommendations': analysis.recommendations
                }
            }
            
            logger.info(f"内存泄漏检测完成: 泄漏检测 {analysis.is_leak_detected}, "
                       f"泄漏率 {analysis.leak_rate:.2f} bytes/s")
            
            return results
            
        except Exception as e:
            logger.error(f"内存泄漏检测测试失败: {e}")
            return {'module': 'memory_leak', 'error': str(e), 'traceback': traceback.format_exc()}
    
    async def run_resource_monitoring_tests(
        self,
        monitoring_duration: int = 30
    ) -> Dict[str, Any]:
        """
        运行资源利用率监控测试
        
        参数:
            monitoring_duration: 监控持续时间（秒）
        
        返回:
            测试结果
        """
        logger.info(f"开始资源利用率监控测试: {monitoring_duration}秒")
        
        try:
            monitor = ResourceUtilizationMonitor()
            
            # 启动监控
            monitor.start_monitoring()
            
            # 模拟一些系统负载
            import math
            import time
            
            start_time = datetime.now()
            while (datetime.now() - start_time).total_seconds() < monitoring_duration:
                # CPU密集型操作
                result = sum(math.sqrt(i) for i in range(10000))
                
                # 内存操作
                temp_data = [i * 2 for i in range(1000)]
                
                await asyncio.sleep(0.1)
            
            # 获取监控结果
            current_snapshot = monitor.get_current_snapshot()
            resource_stats = monitor.get_resource_statistics()
            monitor.stop_monitoring()
            
            results = {
                'module': 'resource_monitoring',
                'current_snapshot': current_snapshot.to_dict() if current_snapshot else {},
                'resource_statistics': resource_stats,
                'summary': {
                    'peak_cpu': resource_stats.get('cpu_stats', {}).get('peak', 0),
                    'peak_memory': resource_stats.get('memory_stats', {}).get('peak', 0),
                    'avg_cpu': resource_stats.get('cpu_stats', {}).get('average', 0),
                    'avg_memory': resource_stats.get('memory_stats', {}).get('average', 0),
                    'monitoring_duration': monitoring_duration,
                    'total_snapshots': resource_stats.get('total_snapshots', 0)
                }
            }
            
            logger.info(f"资源监控测试完成: 峰值CPU {results['summary']['peak_cpu']:.1f}%, "
                       f"峰值内存 {results['summary']['peak_memory']:.1f}%")
            
            return results
            
        except Exception as e:
            logger.error(f"资源利用率监控测试失败: {e}")
            return {'module': 'resource_monitoring', 'error': str(e), 'traceback': traceback.format_exc()}
    
    async def run_execution_efficiency_tests(self) -> Dict[str, Any]:
        """
        运行执行效率测试
        
        返回:
            测试结果
        """
        logger.info("开始执行效率测试")
        
        try:
            tester = ExecutionEfficiencyTester()
            
            # 测试函数执行效率
            def fibonacci(n: int) -> int:
                """斐波那契数列计算函数"""
                if n <= 1:
                    return n
                return fibonacci(n - 1) + fibonacci(n - 2)
            
            def quick_sort(arr: List[int]) -> List[int]:
                """快速排序函数"""
                if len(arr) <= 1:
                    return arr
                pivot = arr[len(arr) // 2]
                left = [x for x in arr if x < pivot]
                middle = [x for x in arr if x == pivot]
                right = [x for x in arr if x > pivot]
                return quick_sort(left) + middle + quick_sort(right)
            
            # 执行函数性能测试
            fib_metrics = tester.measure_function_execution(
                lambda: fibonacci(20)
            )
            
            # 排序性能测试
            import random
            test_array = [random.randint(1, 1000) for _ in range(1000)]
            sort_metrics = tester.measure_function_execution(
                lambda: quick_sort(test_array.copy())
            )
            
            # 内存使用分析
            with tester.profile_memory("主运行器_内存使用分析") as profiler:
                # 模拟内存密集型操作
                large_data = [[i * j for j in range(100)] for i in range(100)]
                processed_data = [sum(row) for row in large_data]
                del large_data, processed_data
            
            memory_profile = profiler.get_profile()
            
            results = {
                'module': 'execution_efficiency',
                'fibonacci_test': fib_metrics.to_dict(),
                'sorting_test': sort_metrics.to_dict(),
                'memory_profile': memory_profile.to_dict(),
                'summary': {
                    'fibonacci_avg_time': fib_metrics.average_execution_time,
                    'sorting_avg_time': sort_metrics.average_execution_time,
                    'fibonacci_performance_grade': fib_metrics.performance_grade,
                    'sorting_performance_grade': sort_metrics.performance_grade,
                    'memory_peak_usage': memory_profile.peak_memory_mb,
                    'memory_efficiency_grade': memory_profile.efficiency_grade
                }
            }
            
            logger.info(f"执行效率测试完成: 斐波那契 {fib_metrics.average_execution_time:.6f}s, "
                       f"排序 {sort_metrics.average_execution_time:.6f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"执行效率测试失败: {e}")
            return {'module': 'execution_efficiency', 'error': str(e), 'traceback': traceback.format_exc()}
    
    async def run_integration_tests(
        self,
        target_url: str = "https://httpbin.org/delay/0.1",
        test_duration: int = 120,
        concurrent_users: int = 20,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        运行集成测试
        
        参数:
            target_url: 测试目标URL
            test_duration: 测试持续时间（秒）
            concurrent_users: 并发用户数
            output_dir: 报告输出目录
        
        返回:
            测试结果
        """
        logger.info(f"开始集成测试: {target_url}, {test_duration}秒, {concurrent_users}用户")
        
        try:
            suite = IntegratedPerformanceTestSuite(self.config)
            
            results = await suite.run_comprehensive_test(
                test_target_url=target_url,
                test_duration=test_duration,
                concurrent_users=concurrent_users,
                memory_check_interval=30,
                report_output_dir=output_dir
            )
            
            # 清理资源
            await suite._cleanup_resources()
            
            logger.info(f"集成测试完成: 总体等级 {results.get('summary', {}).get('overall_grade', 'N/A')}")
            
            return results
            
        except Exception as e:
            logger.error(f"集成测试失败: {e}")
            return {'module': 'integration', 'error': str(e), 'traceback': traceback.format_exc()}
    
    async def run_all_tests(
        self,
        target_url: str = "https://httpbin.org/delay/0.1",
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        运行所有性能测试
        
        参数:
            target_url: 测试目标URL
            output_dir: 报告输出目录
        
        返回:
            所有测试结果
        """
        logger.info("开始运行所有性能测试")
        
        all_results = {
            'test_session': f"all_tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'start_time': self.start_time.isoformat(),
            'target_url': target_url,
            'tests': {}
        }
        
        # 运行各个测试模块
        test_modules = [
            ('response_time', self.run_response_time_tests(target_url)),
            ('concurrency', self.run_concurrency_tests(target_url, 30, 30)),  # 减少负载以加快测试
            ('memory_leak', self.run_memory_leak_tests(60, 15)),
            ('resource_monitoring', self.run_resource_monitoring_tests(30)),
            ('execution_efficiency', self.run_execution_efficiency_tests()),
            ('integration', self.run_integration_tests(target_url, 90, 15, output_dir))
        ]
        
        for module_name, test_coro in test_modules:
            try:
                logger.info(f"运行 {module_name} 测试...")
                result = await test_coro
                all_results['tests'][module_name] = result
                
                # 检查是否有错误
                if 'error' in result:
                    logger.warning(f"{module_name} 测试出现错误: {result['error']}")
                else:
                    logger.info(f"{module_name} 测试完成")
                    
            except Exception as e:
                logger.error(f"{module_name} 测试异常: {e}")
                all_results['tests'][module_name] = {
                    'module': module_name,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
        
        # 生成总体摘要
        all_results['end_time'] = datetime.now().isoformat()
        all_results['total_duration'] = str(datetime.now() - self.start_time)
        all_results['summary'] = self._generate_overall_summary(all_results['tests'])
        
        # 保存结果到文件
        if output_dir:
            await self._save_results(all_results, output_dir)
        
        logger.info(f"所有性能测试完成，总体等级: {all_results['summary']['overall_grade']}")
        
        return all_results
    
    def _generate_overall_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成总体测试摘要"""
        summary = {
            'overall_grade': 'F',
            'total_tests': len(test_results),
            'successful_tests': 0,
            'failed_tests': 0,
            'key_findings': [],
            'recommendations': []
        }
        
        grades = []
        
        for module_name, result in test_results.items():
            if 'error' in result:
                summary['failed_tests'] += 1
                summary['key_findings'].append(f"{module_name}测试失败: {result['error']}")
            else:
                summary['successful_tests'] += 1
                
                # 收集等级信息
                if module_name == 'response_time':
                    sync_grade = result.get('summary', {}).get('sync_grade', 'F')
                    async_grade = result.get('summary', {}).get('async_grade', 'F')
                    grades.extend([sync_grade, async_grade])
                    
                elif module_name == 'concurrency':
                    thread_met = result.get('summary', {}).get('thread_requirements_met', False)
                    async_met = result.get('summary', {}).get('async_requirements_met', False)
                    grades.append('A' if thread_met else 'C')
                    grades.append('A' if async_met else 'C')
                    
                elif module_name == 'execution_efficiency':
                    fib_grade = result.get('summary', {}).get('fibonacci_performance_grade', 'F')
                    sort_grade = result.get('summary', {}).get('sorting_performance_grade', 'F')
                    memory_grade = result.get('summary', {}).get('memory_efficiency_grade', 'F')
                    grades.extend([fib_grade, sort_grade, memory_grade])
                    
                elif module_name == 'integration':
                    overall_grade = result.get('summary', {}).get('overall_grade', 'F')
                    grades.append(overall_grade)
        
        # 计算总体等级
        if grades:
            grade_values = {'A+': 6, 'A': 5, 'B': 4, 'C': 3, 'D': 2, 'F': 1}
            avg_grade_value = sum(grade_values.get(g, 1) for g in grades) / len(grades)
            
            if avg_grade_value >= 5.5:
                summary['overall_grade'] = 'A+'
            elif avg_grade_value >= 4.5:
                summary['overall_grade'] = 'A'
            elif avg_grade_value >= 3.5:
                summary['overall_grade'] = 'B'
            elif avg_grade_value >= 2.5:
                summary['overall_grade'] = 'C'
            elif avg_grade_value >= 1.5:
                summary['overall_grade'] = 'D'
            else:
                summary['overall_grade'] = 'F'
        
        # 生成建议
        if summary['failed_tests'] > 0:
            summary['recommendations'].append("修复失败的测试模块")
        
        if summary['overall_grade'] in ['D', 'F']:
            summary['recommendations'].append("性能需要显著改进")
        elif summary['overall_grade'] in ['B', 'C']:
            summary['recommendations'].append("性能有改进空间")
        
        return summary
    
    async def _save_results(self, results: Dict[str, Any], output_dir: str) -> None:
        """保存测试结果到文件"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 保存JSON结果
            json_file = output_path / f"all_performance_tests_{results['test_session']}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
            # 生成简化的HTML报告
            html_file = output_path / f"performance_summary_{results['test_session']}.html"
            await self._generate_summary_html(results, html_file)
            
            logger.info(f"测试结果已保存: JSON={json_file}, HTML={html_file}")
            
        except Exception as e:
            logger.error(f"保存测试结果失败: {e}")
    
    async def _generate_summary_html(self, results: Dict[str, Any], output_file: Path) -> None:
        """生成简化的HTML摘要报告"""
        summary = results.get('summary', {})
        
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HarborAI性能测试摘要报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ margin: 20px 0; }}
        .grade {{ font-size: 24px; font-weight: bold; }}
        .grade.A {{ color: green; }}
        .grade.B {{ color: orange; }}
        .grade.C {{ color: orange; }}
        .grade.D {{ color: red; }}
        .grade.F {{ color: red; }}
        .test-results {{ margin: 20px 0; }}
        .test-module {{ margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 3px; }}
        .success {{ background-color: #d4edda; }}
        .error {{ background-color: #f8d7da; }}
        .recommendations {{ background-color: #fff3cd; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>HarborAI性能测试摘要报告</h1>
        <p>测试会话: {results['test_session']}</p>
        <p>开始时间: {results['start_time']}</p>
        <p>结束时间: {results['end_time']}</p>
        <p>总持续时间: {results['total_duration']}</p>
        <p>测试目标: {results['target_url']}</p>
    </div>
    
    <div class="summary">
        <h2>总体评估</h2>
        <p class="grade {summary['overall_grade']}">总体等级: {summary['overall_grade']}</p>
        <p>总测试数: {summary['total_tests']}</p>
        <p>成功测试: {summary['successful_tests']}</p>
        <p>失败测试: {summary['failed_tests']}</p>
    </div>
    
    <div class="test-results">
        <h2>测试结果详情</h2>
"""
        
        for module_name, result in results['tests'].items():
            status_class = "error" if "error" in result else "success"
            status_text = "失败" if "error" in result else "成功"
            
            html_content += f"""
        <div class="test-module {status_class}">
            <h3>{module_name} - {status_text}</h3>
"""
            
            if "error" in result:
                html_content += f"<p>错误: {result['error']}</p>"
            else:
                # 添加模块特定的摘要信息
                if 'summary' in result:
                    html_content += "<ul>"
                    for key, value in result['summary'].items():
                        html_content += f"<li>{key}: {value}</li>"
                    html_content += "</ul>"
            
            html_content += "</div>"
        
        html_content += """
    </div>
    
    <div class="recommendations">
        <h2>改进建议</h2>
        <ul>
"""
        
        for recommendation in summary.get('recommendations', []):
            html_content += f"<li>{recommendation}</li>"
        
        html_content += """
        </ul>
    </div>
</body>
</html>
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)


def create_argument_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="HarborAI性能测试主运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python run_performance_tests.py --all --output ./reports
  python run_performance_tests.py --module response_time --url https://api.example.com
  python run_performance_tests.py --integration --duration 300 --users 100
  python run_performance_tests.py --quick-test
        """
    )
    
    # 测试模式选择
    test_group = parser.add_mutually_exclusive_group(required=True)
    test_group.add_argument('--all', action='store_true', help='运行所有性能测试')
    test_group.add_argument('--module', choices=[
        'response_time', 'concurrency', 'memory_leak', 
        'resource_monitoring', 'execution_efficiency'
    ], help='运行指定的测试模块')
    test_group.add_argument('--integration', action='store_true', help='运行集成测试')
    test_group.add_argument('--quick-test', action='store_true', help='运行快速集成测试')
    
    # 测试配置参数
    parser.add_argument('--url', default='https://httpbin.org/delay/0.1', 
                       help='测试目标URL (默认: https://httpbin.org/delay/0.1)')
    parser.add_argument('--users', type=int, default=50, 
                       help='并发用户数 (默认: 50)')
    parser.add_argument('--requests', type=int, default=100, 
                       help='每用户请求数 (默认: 100)')
    parser.add_argument('--duration', type=int, default=120, 
                       help='测试持续时间（秒） (默认: 120)')
    parser.add_argument('--output', help='报告输出目录')
    
    # 日志配置
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='日志级别 (默认: INFO)')
    parser.add_argument('--quiet', action='store_true', help='静默模式，只输出错误')
    
    return parser


async def main():
    """主函数"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # 配置日志级别
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    else:
        logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # 创建性能测试配置
    config = PerformanceConfig(
        test_duration=args.duration,
        max_concurrent_users=args.users,
        response_time_threshold=5.0,
        success_rate_threshold=0.999
    )
    
    # 创建测试运行器
    runner = PerformanceTestRunner(config)
    
    try:
        # 根据参数运行相应的测试
        if args.all:
            results = await runner.run_all_tests(args.url, args.output)
            print(f"\n=== 所有测试完成 ===")
            print(f"总体等级: {results['summary']['overall_grade']}")
            print(f"成功测试: {results['summary']['successful_tests']}/{results['summary']['total_tests']}")
            
        elif args.module:
            if args.module == 'response_time':
                results = await runner.run_response_time_tests(args.url, args.requests)
            elif args.module == 'concurrency':
                results = await runner.run_concurrency_tests(args.url, args.users, args.requests)
            elif args.module == 'memory_leak':
                results = await runner.run_memory_leak_tests(args.duration)
            elif args.module == 'resource_monitoring':
                results = await runner.run_resource_monitoring_tests(args.duration)
            elif args.module == 'execution_efficiency':
                results = await runner.run_execution_efficiency_tests()
            
            print(f"\n=== {args.module} 测试完成 ===")
            if 'error' in results:
                print(f"测试失败: {results['error']}")
            else:
                print(f"测试成功，详细结果请查看日志")
                
        elif args.integration:
            results = await runner.run_integration_tests(
                args.url, args.duration, args.users, args.output
            )
            print(f"\n=== 集成测试完成 ===")
            if 'error' in results:
                print(f"测试失败: {results['error']}")
            else:
                summary = results.get('summary', {})
                print(f"总体等级: {summary.get('overall_grade', 'N/A')}")
                
        elif args.quick_test:
            results = await run_quick_integration_test(args.url, args.output)
            print(f"\n=== 快速集成测试完成 ===")
            summary = results.get('summary', {})
            print(f"总体等级: {summary.get('overall_grade', 'N/A')}")
        
        # 输出报告位置
        if args.output:
            print(f"\n报告已保存到: {args.output}")
            
    except KeyboardInterrupt:
        logger.info("测试被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"测试执行异常: {e}")
        if not args.quiet:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # 运行主函数
    asyncio.run(main())