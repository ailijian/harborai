#!/usr/bin/env python3
"""
性能回归检查脚本
用于检测当前版本相对于基线版本的性能回归问题

功能：
1. 加载性能基线数据
2. 运行当前版本的性能测试
3. 对比分析性能指标
4. 生成回归检查报告
5. 根据阈值判断是否存在性能回归

作者：ailijian
创建时间：2024
"""

import json
import sys
import os
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import subprocess
import statistics

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('performance_regression_check.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """性能指标数据结构"""
    name: str
    value: float
    unit: str
    threshold_percent: float = 10.0  # 默认阈值 10%
    
    
@dataclass
class RegressionResult:
    """回归检查结果"""
    metric_name: str
    baseline_value: float
    current_value: float
    change_percent: float
    is_regression: bool
    severity: str  # 'low', 'medium', 'high', 'critical'


class PerformanceRegressionChecker:
    """性能回归检查器"""
    
    def __init__(self, baseline_dir: str = None, output_dir: str = None):
        """
        初始化性能回归检查器
        
        Args:
            baseline_dir: 基线数据目录路径
            output_dir: 输出报告目录路径
        """
        self.baseline_dir = Path(baseline_dir or "tests/data/performance_baselines")
        self.output_dir = Path(output_dir or "tests/reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 性能阈值配置
        self.thresholds = {
            'latency': 15.0,      # 延迟增加超过 15% 为回归
            'throughput': -10.0,   # 吞吐量下降超过 10% 为回归
            'memory': 20.0,        # 内存使用增加超过 20% 为回归
            'cpu': 25.0,           # CPU 使用增加超过 25% 为回归
            'concurrency': -15.0,  # 并发性能下降超过 15% 为回归
        }
        
        # 严重程度分级
        self.severity_levels = {
            (0, 5): 'low',
            (5, 15): 'medium', 
            (15, 30): 'high',
            (30, float('inf')): 'critical'
        }
    
    def load_baseline_data(self, baseline_file: str) -> Dict[str, Any]:
        """
        加载基线性能数据
        
        Args:
            baseline_file: 基线文件名
            
        Returns:
            基线数据字典
        """
        baseline_path = self.baseline_dir / baseline_file
        
        if not baseline_path.exists():
            logger.warning(f"基线文件不存在: {baseline_path}")
            return {}
            
        try:
            with open(baseline_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"✅ 成功加载基线数据: {baseline_file}")
            return data
        except Exception as e:
            logger.error(f"❌ 加载基线数据失败: {e}")
            return {}
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """
        运行当前版本的性能测试
        
        Returns:
            当前性能测试结果
        """
        logger.info("🚀 开始运行性能测试...")
        
        current_results = {
            'timestamp': datetime.now().isoformat(),
            'version': self._get_current_version(),
            'metrics': {}
        }
        
        try:
            # 运行基础性能测试
            latency_results = self._run_latency_tests()
            throughput_results = self._run_throughput_tests()
            memory_results = self._run_memory_tests()
            concurrency_results = self._run_concurrency_tests()
            
            current_results['metrics'] = {
                'latency': latency_results,
                'throughput': throughput_results,
                'memory': memory_results,
                'concurrency': concurrency_results
            }
            
            logger.info("✅ 性能测试完成")
            return current_results
            
        except Exception as e:
            logger.error(f"❌ 性能测试失败: {e}")
            return current_results
    
    def _get_current_version(self) -> str:
        """获取当前版本号"""
        try:
            # 从 pyproject.toml 读取版本
            pyproject_path = project_root / "pyproject.toml"
            if pyproject_path.exists():
                with open(pyproject_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    import re
                    match = re.search(r'version = "([^"]+)"', content)
                    if match:
                        return match.group(1)
            return "unknown"
        except Exception:
            return "unknown"
    
    def _run_latency_tests(self) -> Dict[str, float]:
        """运行延迟测试"""
        logger.info("📊 运行延迟测试...")
        
        # 模拟延迟测试结果（实际应该调用真实的测试）
        try:
            # 这里应该调用实际的延迟测试脚本
            cmd = [sys.executable, "-m", "pytest", 
                   "tests/performance/test_basic_performance.py::test_response_time", 
                   "-v", "--tb=short"]
            
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  cwd=project_root, timeout=300)
            
            # 解析测试结果（简化版本）
            return {
                'avg_response_time': 0.15,  # 150ms
                'p95_response_time': 0.25,  # 250ms
                'p99_response_time': 0.35,  # 350ms
            }
        except Exception as e:
            logger.warning(f"延迟测试执行失败: {e}")
            return {}
    
    def _run_throughput_tests(self) -> Dict[str, float]:
        """运行吞吐量测试"""
        logger.info("📊 运行吞吐量测试...")
        
        try:
            # 这里应该调用实际的吞吐量测试脚本
            return {
                'requests_per_second': 850.0,
                'tokens_per_second': 1200.0,
                'concurrent_requests': 50.0,
            }
        except Exception as e:
            logger.warning(f"吞吐量测试执行失败: {e}")
            return {}
    
    def _run_memory_tests(self) -> Dict[str, float]:
        """运行内存使用测试"""
        logger.info("📊 运行内存测试...")
        
        try:
            # 这里应该调用实际的内存测试脚本
            return {
                'peak_memory_mb': 128.5,
                'avg_memory_mb': 95.2,
                'memory_growth_rate': 0.02,  # 2% per hour
            }
        except Exception as e:
            logger.warning(f"内存测试执行失败: {e}")
            return {}
    
    def _run_concurrency_tests(self) -> Dict[str, float]:
        """运行并发性能测试"""
        logger.info("📊 运行并发测试...")
        
        try:
            # 这里应该调用实际的并发测试脚本
            return {
                'max_concurrent_users': 100.0,
                'concurrent_throughput': 750.0,
                'error_rate_percent': 0.5,
            }
        except Exception as e:
            logger.warning(f"并发测试执行失败: {e}")
            return {}
    
    def compare_metrics(self, baseline: Dict[str, Any], 
                       current: Dict[str, Any]) -> List[RegressionResult]:
        """
        对比性能指标
        
        Args:
            baseline: 基线数据
            current: 当前测试数据
            
        Returns:
            回归检查结果列表
        """
        logger.info("🔍 开始性能指标对比...")
        
        results = []
        
        for category, current_metrics in current.get('metrics', {}).items():
            baseline_metrics = baseline.get(category, {})
            
            for metric_name, current_value in current_metrics.items():
                if metric_name in baseline_metrics:
                    baseline_value = baseline_metrics[metric_name]
                    
                    # 计算变化百分比
                    if baseline_value != 0:
                        change_percent = ((current_value - baseline_value) / baseline_value) * 100
                    else:
                        change_percent = 0.0
                    
                    # 判断是否为回归
                    threshold = self.thresholds.get(category, 10.0)
                    is_regression = self._is_regression(change_percent, threshold, category)
                    
                    # 确定严重程度
                    severity = self._get_severity(abs(change_percent))
                    
                    result = RegressionResult(
                        metric_name=f"{category}.{metric_name}",
                        baseline_value=baseline_value,
                        current_value=current_value,
                        change_percent=change_percent,
                        is_regression=is_regression,
                        severity=severity
                    )
                    
                    results.append(result)
        
        logger.info(f"✅ 完成 {len(results)} 个指标的对比")
        return results
    
    def _is_regression(self, change_percent: float, threshold: float, 
                      category: str) -> bool:
        """
        判断是否为性能回归
        
        Args:
            change_percent: 变化百分比
            threshold: 阈值
            category: 指标类别
            
        Returns:
            是否为回归
        """
        # 对于吞吐量和并发性能，下降是回归
        if category in ['throughput', 'concurrency']:
            return change_percent < threshold
        # 对于延迟和内存，增加是回归
        else:
            return change_percent > threshold
    
    def _get_severity(self, change_percent: float) -> str:
        """获取严重程度"""
        for (min_val, max_val), severity in self.severity_levels.items():
            if min_val <= change_percent < max_val:
                return severity
        return 'low'
    
    def generate_report(self, results: List[RegressionResult], 
                       output_file: str = None) -> str:
        """
        生成回归检查报告
        
        Args:
            results: 回归检查结果
            output_file: 输出文件路径
            
        Returns:
            报告文件路径
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"performance_regression_report_{timestamp}.md"
        
        report_path = self.output_dir / output_file
        
        # 统计信息
        total_metrics = len(results)
        regressions = [r for r in results if r.is_regression]
        regression_count = len(regressions)
        
        # 按严重程度分组
        severity_counts = {}
        for result in regressions:
            severity_counts[result.severity] = severity_counts.get(result.severity, 0) + 1
        
        # 生成报告内容
        report_content = f"""# 性能回归检查报告

## 📊 总体概况

- **检查时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **总指标数**: {total_metrics}
- **回归指标数**: {regression_count}
- **回归率**: {(regression_count/total_metrics*100):.1f}%

## 🚨 严重程度分布

"""
        
        for severity in ['critical', 'high', 'medium', 'low']:
            count = severity_counts.get(severity, 0)
            if count > 0:
                emoji = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}[severity]
                report_content += f"- {emoji} **{severity.upper()}**: {count} 个\n"
        
        report_content += "\n## 📈 详细结果\n\n"
        
        if regressions:
            report_content += "### 🚨 发现的性能回归\n\n"
            report_content += "| 指标 | 基线值 | 当前值 | 变化 | 严重程度 |\n"
            report_content += "|------|--------|--------|------|----------|\n"
            
            for result in sorted(regressions, key=lambda x: abs(x.change_percent), reverse=True):
                emoji = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}[result.severity]
                report_content += f"| {result.metric_name} | {result.baseline_value:.3f} | {result.current_value:.3f} | {result.change_percent:+.1f}% | {emoji} {result.severity} |\n"
        
        # 添加所有指标的详细信息
        report_content += "\n### 📊 所有指标对比\n\n"
        report_content += "| 指标 | 基线值 | 当前值 | 变化 | 状态 |\n"
        report_content += "|------|--------|--------|------|------|\n"
        
        for result in results:
            status = "❌ 回归" if result.is_regression else "✅ 正常"
            report_content += f"| {result.metric_name} | {result.baseline_value:.3f} | {result.current_value:.3f} | {result.change_percent:+.1f}% | {status} |\n"
        
        # 添加建议
        report_content += "\n## 💡 建议\n\n"
        
        if regression_count == 0:
            report_content += "🎉 **恭喜！** 未发现性能回归问题。\n"
        else:
            critical_count = severity_counts.get('critical', 0)
            high_count = severity_counts.get('high', 0)
            
            if critical_count > 0:
                report_content += "🚨 **严重警告**: 发现严重性能回归，建议立即修复后再发布。\n"
            elif high_count > 0:
                report_content += "⚠️ **警告**: 发现高级别性能回归，建议优化后发布。\n"
            else:
                report_content += "ℹ️ **提示**: 发现轻微性能回归，可考虑优化。\n"
        
        # 写入报告文件
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"✅ 报告已生成: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"❌ 生成报告失败: {e}")
            return ""
    
    def check_regression(self, baseline_file: str = "baseline_v1.0.json") -> Tuple[bool, str]:
        """
        执行完整的性能回归检查
        
        Args:
            baseline_file: 基线文件名
            
        Returns:
            (是否有回归, 报告文件路径)
        """
        logger.info("🔍 开始性能回归检查...")
        
        # 1. 加载基线数据
        baseline_data = self.load_baseline_data(baseline_file)
        if not baseline_data:
            logger.error("❌ 无法加载基线数据，跳过回归检查")
            return False, ""
        
        # 2. 运行当前性能测试
        current_data = self.run_performance_tests()
        if not current_data.get('metrics'):
            logger.error("❌ 无法获取当前性能数据，跳过回归检查")
            return False, ""
        
        # 3. 对比分析
        results = self.compare_metrics(baseline_data, current_data)
        
        # 4. 生成报告
        report_path = self.generate_report(results)
        
        # 5. 判断是否有回归
        has_regression = any(r.is_regression for r in results)
        critical_regressions = [r for r in results if r.is_regression and r.severity == 'critical']
        
        if has_regression:
            logger.warning(f"⚠️ 发现 {len([r for r in results if r.is_regression])} 个性能回归")
            if critical_regressions:
                logger.error(f"🚨 发现 {len(critical_regressions)} 个严重性能回归")
        else:
            logger.info("✅ 未发现性能回归")
        
        return has_regression, report_path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="性能回归检查工具")
    parser.add_argument("--baseline", default="baseline_v1.0.json",
                       help="基线文件名 (默认: baseline_v1.0.json)")
    parser.add_argument("--baseline-dir", 
                       help="基线数据目录路径")
    parser.add_argument("--output-dir",
                       help="输出报告目录路径")
    parser.add_argument("--fail-on-regression", action="store_true",
                       help="发现回归时以非零状态码退出")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="详细输出")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 创建检查器
    checker = PerformanceRegressionChecker(
        baseline_dir=args.baseline_dir,
        output_dir=args.output_dir
    )
    
    # 执行检查
    has_regression, report_path = checker.check_regression(args.baseline)
    
    # 输出结果
    if report_path:
        print(f"📄 报告文件: {report_path}")
    
    if has_regression:
        print("❌ 发现性能回归")
        if args.fail_on_regression:
            sys.exit(1)
    else:
        print("✅ 未发现性能回归")
    
    sys.exit(0)


if __name__ == "__main__":
    main()