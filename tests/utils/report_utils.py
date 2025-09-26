#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
报告生成工具模块

功能：提供测试报告生成、指标收集、Prometheus导出等功能
作者：HarborAI测试团队
创建时间：2024年
"""

import json
import time
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
import os
import sys
import traceback
from jinja2 import Template, Environment, FileSystemLoader
from prometheus_client import Counter as PrometheusCounter, Histogram, Gauge, start_http_server, CollectorRegistry
import psutil


@dataclass
class TestSummary:
    """测试摘要数据类
    
    功能：存储测试统计信息
    """
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    error_tests: int = 0
    execution_time: float = 0.0
    coverage_percentage: float = 0.0
    success_rate: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    environment: str = "unknown"
    test_files: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """计算成功率"""
        if self.total_tests > 0:
            self.success_rate = (self.passed_tests / self.total_tests) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['start_time'] = self.start_time.isoformat() if self.start_time else None
        data['end_time'] = self.end_time.isoformat() if self.end_time else None
        return data


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    avg_response_time: float = 0.0
    max_response_time: float = 0.0
    min_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    concurrent_requests: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class SecurityMetrics:
    """安全指标数据类"""
    vulnerabilities_found: int = 0
    critical_issues: int = 0
    high_issues: int = 0
    medium_issues: int = 0
    low_issues: int = 0
    security_score: float = 100.0
    scan_duration: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


class ReportGenerator:
    """报告生成器
    
    功能：生成各种格式的测试报告
    """
    
    def __init__(self, template_dir: str = None):
        """初始化报告生成器
        
        参数：
            template_dir: 模板目录路径
        """
        self.template_dir = template_dir or str(Path(__file__).parent / "templates")
        self.env = None
        self._setup_templates()
        
        # 报告数据
        self.test_summary = TestSummary()
        self.performance_metrics = PerformanceMetrics()
        self.security_metrics = SecurityMetrics()
        self.test_results = []
        self.error_logs = []
        
        logging.info("报告生成器初始化完成")
    
    def _setup_templates(self):
        """设置模板环境"""
        try:
            # 确保模板目录存在
            Path(self.template_dir).mkdir(parents=True, exist_ok=True)
            
            # 创建默认模板
            self._create_default_templates()
            
            # 设置Jinja2环境
            self.env = Environment(
                loader=FileSystemLoader(self.template_dir),
                autoescape=True
            )
            
        except Exception as e:
            logging.warning(f"模板设置失败：{e}，将使用内置模板")
            self.env = Environment()
    
    def _create_default_templates(self):
        """创建默认模板文件"""
        templates = {
            "summary_report.md": self._get_summary_template(),
            "performance_report.html": self._get_performance_template(),
            "security_report.html": self._get_security_template()
        }
        
        for filename, content in templates.items():
            template_path = Path(self.template_dir) / filename
            if not template_path.exists():
                template_path.write_text(content, encoding='utf-8')
    
    def _get_summary_template(self) -> str:
        """获取摘要报告模板"""
        return """
# 测试报告摘要

## 测试概览

| 指标 | 数值 |
|------|------|
| 总测试数 | {{ summary.total_tests }} |
| 通过测试 | {{ summary.passed_tests }} |
| 失败测试 | {{ summary.failed_tests }} |
| 跳过测试 | {{ summary.skipped_tests }} |
| 错误测试 | {{ summary.error_tests }} |
| 成功率 | {{ "%.2f" | format(summary.success_rate) }}% |
| 执行时间 | {{ "%.2f" | format(summary.execution_time) }}秒 |
| 代码覆盖率 | {{ "%.2f" | format(summary.coverage_percentage) }}% |

## 测试状态

{% if summary.success_rate >= 95 %}
✅ **测试状态：优秀** - 成功率达到{{ "%.2f" | format(summary.success_rate) }}%
{% elif summary.success_rate >= 80 %}
⚠️ **测试状态：良好** - 成功率为{{ "%.2f" | format(summary.success_rate) }}%，建议关注失败用例
{% else %}
❌ **测试状态：需要改进** - 成功率仅为{{ "%.2f" | format(summary.success_rate) }}%，请及时修复问题
{% endif %}

## 详细报告

- [性能测试报告](performance_report.html)
- [安全测试报告](security_report.html)

---
生成时间：{{ timestamp }}
环境：{{ summary.environment }}
"""
    
    def _get_performance_template(self) -> str:
        """获取性能报告模板"""
        return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>性能测试报告</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f5f5f5; padding: 20px; border-radius: 5px; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
        .metric-card { background: white; border: 1px solid #ddd; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric-value { font-size: 24px; font-weight: bold; color: #2196F3; }
        .metric-label { color: #666; margin-top: 5px; }
        .chart-container { margin: 20px 0; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>性能测试报告</h1>
        <p>生成时间：{{ timestamp }}</p>
        <p>测试环境：{{ environment }}</p>
    </div>
    
    <div class="metrics">
        <div class="metric-card">
            <div class="metric-value">{{ "%.2f" | format(performance.avg_response_time) }}ms</div>
            <div class="metric-label">平均响应时间</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{{ "%.2f" | format(performance.max_response_time) }}ms</div>
            <div class="metric-label">最大响应时间</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{{ "%.2f" | format(performance.throughput) }}</div>
            <div class="metric-label">吞吐量 (req/s)</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{{ "%.2f" | format(performance.error_rate) }}%</div>
            <div class="metric-label">错误率</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{{ "%.2f" | format(performance.cpu_usage) }}%</div>
            <div class="metric-label">CPU使用率</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{{ "%.2f" | format(performance.memory_usage) }}MB</div>
            <div class="metric-label">内存使用</div>
        </div>
    </div>
    
    <h2>性能分析</h2>
    <table>
        <tr>
            <th>指标</th>
            <th>数值</th>
            <th>状态</th>
            <th>建议</th>
        </tr>
        <tr>
            <td>平均响应时间</td>
            <td>{{ "%.2f" | format(performance.avg_response_time) }}ms</td>
            <td>{% if performance.avg_response_time < 100 %}✅ 优秀{% elif performance.avg_response_time < 500 %}⚠️ 良好{% else %}❌ 需要优化{% endif %}</td>
            <td>{% if performance.avg_response_time >= 500 %}建议优化代码逻辑和数据库查询{% endif %}</td>
        </tr>
        <tr>
            <td>错误率</td>
            <td>{{ "%.2f" | format(performance.error_rate) }}%</td>
            <td>{% if performance.error_rate < 1 %}✅ 优秀{% elif performance.error_rate < 5 %}⚠️ 良好{% else %}❌ 需要修复{% endif %}</td>
            <td>{% if performance.error_rate >= 5 %}请检查错误日志并修复相关问题{% endif %}</td>
        </tr>
    </table>
</body>
</html>
"""
    
    def _get_security_template(self) -> str:
        """获取安全报告模板"""
        return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>安全测试报告</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f5f5f5; padding: 20px; border-radius: 5px; }
        .security-score { font-size: 48px; font-weight: bold; text-align: center; margin: 20px 0; }
        .score-excellent { color: #4CAF50; }
        .score-good { color: #FF9800; }
        .score-poor { color: #F44336; }
        .vulnerabilities { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin: 20px 0; }
        .vuln-card { background: white; border: 1px solid #ddd; padding: 15px; border-radius: 5px; text-align: center; }
        .vuln-critical { border-left: 5px solid #F44336; }
        .vuln-high { border-left: 5px solid #FF9800; }
        .vuln-medium { border-left: 5px solid #FFC107; }
        .vuln-low { border-left: 5px solid #4CAF50; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>安全测试报告</h1>
        <p>生成时间：{{ timestamp }}</p>
        <p>扫描时长：{{ "%.2f" | format(security.scan_duration) }}秒</p>
    </div>
    
    <div class="security-score {% if security.security_score >= 80 %}score-excellent{% elif security.security_score >= 60 %}score-good{% else %}score-poor{% endif %}">
        安全评分：{{ "%.1f" | format(security.security_score) }}/100
    </div>
    
    <div class="vulnerabilities">
        <div class="vuln-card vuln-critical">
            <h3>{{ security.critical_issues }}</h3>
            <p>严重漏洞</p>
        </div>
        <div class="vuln-card vuln-high">
            <h3>{{ security.high_issues }}</h3>
            <p>高危漏洞</p>
        </div>
        <div class="vuln-card vuln-medium">
            <h3>{{ security.medium_issues }}</h3>
            <p>中危漏洞</p>
        </div>
        <div class="vuln-card vuln-low">
            <h3>{{ security.low_issues }}</h3>
            <p>低危漏洞</p>
        </div>
    </div>
    
    <h2>安全建议</h2>
    <ul>
        {% if security.critical_issues > 0 %}
        <li><strong>紧急：</strong>发现{{ security.critical_issues }}个严重安全漏洞，请立即修复</li>
        {% endif %}
        {% if security.high_issues > 0 %}
        <li><strong>重要：</strong>发现{{ security.high_issues }}个高危漏洞，建议优先处理</li>
        {% endif %}
        {% if security.vulnerabilities_found == 0 %}
        <li>✅ 未发现明显安全漏洞，系统安全性良好</li>
        {% endif %}
        <li>建议定期进行安全扫描和渗透测试</li>
        <li>确保所有输入都经过适当的验证和清理</li>
        <li>使用最新的安全补丁和依赖版本</li>
    </ul>
</body>
</html>
"""
    
    def generate_summary_report(
        self,
        test_results: List[Dict],
        output_file: str = None
    ) -> str:
        """生成测试摘要报告
        
        功能：创建Markdown格式的测试摘要
        参数：
            test_results: 测试结果列表
            output_file: 输出文件路径
        返回：报告内容
        """
        try:
            # 更新测试摘要
            self._update_test_summary(test_results)
            
            # 渲染模板
            template = self.env.get_template("summary_report.md")
            content = template.render(
                summary=self.test_summary,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            
            # 保存文件
            if output_file:
                Path(output_file).write_text(content, encoding='utf-8')
                logging.info(f"摘要报告已保存到：{output_file}")
            
            return content
            
        except Exception as e:
            logging.error(f"生成摘要报告失败：{e}")
            return f"报告生成失败：{e}"
    
    def generate_performance_report(
        self,
        performance_data: Dict,
        output_file: str = None
    ) -> str:
        """生成性能测试报告
        
        功能：创建HTML格式的性能报告
        参数：
            performance_data: 性能数据
            output_file: 输出文件路径
        返回：报告文件路径
        """
        try:
            # 更新性能指标
            self._update_performance_metrics(performance_data)
            
            # 渲染模板
            template = self.env.get_template("performance_report.html")
            content = template.render(
                performance=self.performance_metrics,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                environment=self.test_summary.environment
            )
            
            # 保存文件
            if not output_file:
                output_file = f"performance_report_{int(time.time())}.html"
            
            Path(output_file).write_text(content, encoding='utf-8')
            logging.info(f"性能报告已保存到：{output_file}")
            
            return output_file
            
        except Exception as e:
            logging.error(f"生成性能报告失败：{e}")
            return ""
    
    def generate_security_report(
        self,
        security_data: Dict,
        output_file: str = None
    ) -> str:
        """生成安全测试报告
        
        功能：创建HTML格式的安全报告
        参数：
            security_data: 安全数据
            output_file: 输出文件路径
        返回：报告文件路径
        """
        try:
            # 更新安全指标
            self._update_security_metrics(security_data)
            
            # 渲染模板
            template = self.env.get_template("security_report.html")
            content = template.render(
                security=self.security_metrics,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            
            # 保存文件
            if not output_file:
                output_file = f"security_report_{int(time.time())}.html"
            
            Path(output_file).write_text(content, encoding='utf-8')
            logging.info(f"安全报告已保存到：{output_file}")
            
            return output_file
            
        except Exception as e:
            logging.error(f"生成安全报告失败：{e}")
            return ""
    
    def _update_test_summary(self, test_results: List[Dict]):
        """更新测试摘要"""
        if not test_results:
            return
        
        self.test_summary.total_tests = len(test_results)
        self.test_summary.passed_tests = sum(1 for r in test_results if r.get('status') == 'passed')
        self.test_summary.failed_tests = sum(1 for r in test_results if r.get('status') == 'failed')
        self.test_summary.skipped_tests = sum(1 for r in test_results if r.get('status') == 'skipped')
        self.test_summary.error_tests = sum(1 for r in test_results if r.get('status') == 'error')
        
        # 计算执行时间
        durations = [r.get('duration', 0) for r in test_results if r.get('duration')]
        self.test_summary.execution_time = sum(durations)
        
        # 计算成功率
        if self.test_summary.total_tests > 0:
            self.test_summary.success_rate = (self.test_summary.passed_tests / self.test_summary.total_tests) * 100
        
        # 提取测试文件
        test_files = set(r.get('file', '') for r in test_results if r.get('file'))
        self.test_summary.test_files = list(test_files)
    
    def _update_performance_metrics(self, performance_data: Dict):
        """更新性能指标"""
        for key, value in performance_data.items():
            if hasattr(self.performance_metrics, key):
                setattr(self.performance_metrics, key, value)
    
    def _update_security_metrics(self, security_data: Dict):
        """更新安全指标"""
        for key, value in security_data.items():
            if hasattr(self.security_metrics, key):
                setattr(self.security_metrics, key, value)


class MetricsCollector:
    """指标收集器
    
    功能：收集和存储测试指标数据
    """
    
    def __init__(self, output_dir: str = "metrics"):
        """初始化指标收集器
        
        参数：
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 指标数据
        self.test_metrics = []
        self.performance_metrics = []
        self.security_metrics = []
        self.system_metrics = []
        
        # 线程锁
        self._lock = threading.Lock()
        
        logging.info(f"指标收集器初始化完成，输出目录：{self.output_dir}")
    
    def collect_test_metrics(
        self,
        session_id: str,
        test_results: List[Dict],
        environment_info: Dict = None
    ) -> Dict[str, Any]:
        """收集测试指标
        
        功能：收集测试执行的各项指标
        参数：
            session_id: 测试会话ID
            test_results: 测试结果列表
            environment_info: 环境信息
        返回：收集的指标数据
        """
        try:
            timestamp = datetime.now()
            
            # 基础统计
            total_tests = len(test_results)
            passed_tests = sum(1 for r in test_results if r.get('status') == 'passed')
            failed_tests = sum(1 for r in test_results if r.get('status') == 'failed')
            skipped_tests = sum(1 for r in test_results if r.get('status') == 'skipped')
            error_tests = sum(1 for r in test_results if r.get('status') == 'error')
            
            # 性能统计
            durations = [r.get('duration', 0) for r in test_results if r.get('duration')]
            total_duration = sum(durations)
            avg_duration = total_duration / len(durations) if durations else 0
            max_duration = max(durations) if durations else 0
            min_duration = min(durations) if durations else 0
            
            # 覆盖率统计
            coverage_data = [r.get('coverage', 0) for r in test_results if r.get('coverage')]
            avg_coverage = sum(coverage_data) / len(coverage_data) if coverage_data else 0
            
            # 构建指标数据
            metrics = {
                'session_id': session_id,
                'timestamp': timestamp.isoformat(),
                'environment': environment_info or {},
                'test_results': {
                    'total': total_tests,
                    'passed': passed_tests,
                    'failed': failed_tests,
                    'skipped': skipped_tests,
                    'error': error_tests,
                    'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0
                },
                'performance_metrics': {
                    'total_duration': total_duration,
                    'avg_duration': avg_duration,
                    'max_duration': max_duration,
                    'min_duration': min_duration,
                    'throughput': total_tests / total_duration if total_duration > 0 else 0
                },
                'coverage_metrics': {
                    'avg_coverage': avg_coverage,
                    'coverage_data': coverage_data
                }
            }
            
            # 线程安全地添加到列表
            with self._lock:
                self.test_metrics.append(metrics)
            
            logging.info(f"已收集测试指标，会话ID：{session_id}")
            return metrics
            
        except Exception as e:
            logging.error(f"收集测试指标失败：{e}")
            return {}
    
    def save_metrics(self, filename: str = None) -> str:
        """保存指标到文件
        
        功能：将收集的指标保存为JSON文件
        参数：
            filename: 文件名（可选）
        返回：保存的文件路径
        """
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"metrics_{timestamp}.json"
            
            filepath = self.output_dir / filename
            
            # 合并所有指标
            all_metrics = {
                'test_metrics': self.test_metrics,
                'performance_metrics': self.performance_metrics,
                'security_metrics': self.security_metrics,
                'system_metrics': self.system_metrics,
                'generated_at': datetime.now().isoformat()
            }
            
            # 保存到文件
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(all_metrics, f, ensure_ascii=False, indent=2)
            
            logging.info(f"指标已保存到：{filepath}")
            return str(filepath)
            
        except Exception as e:
            logging.error(f"保存指标失败：{e}")
            return ""
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """获取汇总统计信息
        
        功能：计算所有收集指标的汇总统计
        返回：汇总统计数据
        """
        try:
            if not self.test_metrics:
                return {}
            
            # 计算测试统计
            total_sessions = len(self.test_metrics)
            total_tests = sum(m['test_results']['total'] for m in self.test_metrics)
            total_passed = sum(m['test_results']['passed'] for m in self.test_metrics)
            total_failed = sum(m['test_results']['failed'] for m in self.test_metrics)
            
            # 计算性能统计
            all_durations = []
            for m in self.test_metrics:
                all_durations.append(m['performance_metrics']['total_duration'])
            
            avg_session_duration = sum(all_durations) / len(all_durations) if all_durations else 0
            
            # 计算成功率趋势
            success_rates = [m['test_results']['success_rate'] for m in self.test_metrics]
            avg_success_rate = sum(success_rates) / len(success_rates) if success_rates else 0
            
            return {
                'summary': {
                    'total_sessions': total_sessions,
                    'total_tests': total_tests,
                    'total_passed': total_passed,
                    'total_failed': total_failed,
                    'overall_success_rate': (total_passed / total_tests * 100) if total_tests > 0 else 0
                },
                'performance': {
                    'avg_session_duration': avg_session_duration,
                    'total_execution_time': sum(all_durations)
                },
                'trends': {
                    'avg_success_rate': avg_success_rate,
                    'success_rate_trend': success_rates[-5:] if len(success_rates) >= 5 else success_rates
                }
            }
            
        except Exception as e:
            logging.error(f"计算汇总统计失败：{e}")
            return {}


class TestMetricsExporter:
    """测试指标导出器（Prometheus集成）
    
    功能：将测试指标导出到Prometheus监控系统
    """
    
    def __init__(self, port: int = 8000, registry: CollectorRegistry = None):
        """初始化指标导出器
        
        参数：
            port: HTTP服务端口
            registry: Prometheus注册表
        """
        self.port = port
        self.registry = registry or CollectorRegistry()
        self.server_started = False
        
        # 定义指标
        self._setup_metrics()
        
        logging.info(f"Prometheus指标导出器初始化完成，端口：{port}")
    
    def _setup_metrics(self):
        """设置Prometheus指标"""
        # 测试计数器
        self.test_counter = PrometheusCounter(
            'harborai_tests_total',
            'Total number of tests executed',
            ['status', 'test_type'],
            registry=self.registry
        )
        
        # 测试持续时间
        self.test_duration = Histogram(
            'harborai_test_duration_seconds',
            'Test execution duration in seconds',
            ['test_name', 'test_type'],
            registry=self.registry
        )
        
        # API请求持续时间
        self.api_request_duration = Histogram(
            'harborai_api_request_duration_seconds',
            'API request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # API请求计数器
        self.api_request_counter = PrometheusCounter(
            'harborai_api_requests_total',
            'Total number of API requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        # 并发请求数
        self.concurrent_requests = Gauge(
            'harborai_concurrent_requests',
            'Number of concurrent requests',
            registry=self.registry
        )
        
        # 内存使用
        self.memory_usage = Gauge(
            'harborai_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        # CPU使用率
        self.cpu_usage = Gauge(
            'harborai_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
    
    def start_server(self):
        """启动HTTP服务器
        
        功能：启动Prometheus指标暴露服务器
        """
        try:
            if not self.server_started:
                start_http_server(self.port, registry=self.registry)
                self.server_started = True
                logging.info(f"Prometheus指标服务器已启动，端口：{self.port}")
                logging.info(f"指标访问地址：http://localhost:{self.port}/metrics")
        except Exception as e:
            logging.error(f"启动Prometheus服务器失败：{e}")
    
    def record_test_execution(
        self,
        test_name: str,
        test_type: str,
        status: str,
        duration: float
    ):
        """记录测试执行
        
        功能：记录单个测试的执行情况
        参数：
            test_name: 测试名称
            test_type: 测试类型
            status: 测试状态
            duration: 执行时长
        """
        try:
            # 更新计数器
            self.test_counter.labels(status=status, test_type=test_type).inc()
            
            # 记录持续时间
            self.test_duration.labels(test_name=test_name, test_type=test_type).observe(duration)
            
        except Exception as e:
            logging.error(f"记录测试执行失败：{e}")
    
    def record_api_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float
    ):
        """记录API请求
        
        功能：记录API请求的执行情况
        参数：
            method: HTTP方法
            endpoint: API端点
            status_code: 状态码
            duration: 请求时长
        """
        try:
            # 更新计数器
            self.api_request_counter.labels(
                method=method,
                endpoint=endpoint,
                status_code=str(status_code)
            ).inc()
            
            # 记录持续时间
            self.api_request_duration.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
        except Exception as e:
            logging.error(f"记录API请求失败：{e}")
    
    def update_system_metrics(self):
        """更新系统指标
        
        功能：更新系统资源使用指标
        """
        try:
            # 获取系统信息
            process = psutil.Process()
            
            # 更新内存使用
            memory_info = process.memory_info()
            self.memory_usage.set(memory_info.rss)
            
            # 更新CPU使用率
            cpu_percent = process.cpu_percent()
            self.cpu_usage.set(cpu_percent)
            
        except Exception as e:
            logging.error(f"更新系统指标失败：{e}")
    
    def set_concurrent_requests(self, count: int):
        """设置并发请求数
        
        功能：更新当前并发请求数量
        参数：
            count: 并发请求数
        """
        try:
            self.concurrent_requests.set(count)
        except Exception as e:
            logging.error(f"设置并发请求数失败：{e}")


# 全局实例
report_generator = ReportGenerator()
metrics_collector = MetricsCollector()
metrics_exporter = TestMetricsExporter()


# 便捷函数
def generate_test_report(
    test_results: List[Dict],
    performance_data: Dict = None,
    security_data: Dict = None,
    output_dir: str = "reports"
) -> Dict[str, str]:
    """便捷的测试报告生成函数
    
    功能：一次性生成所有类型的测试报告
    参数：
        test_results: 测试结果列表
        performance_data: 性能数据
        security_data: 安全数据
        output_dir: 输出目录
    返回：生成的报告文件路径字典
    """
    try:
        # 确保输出目录存在
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        reports = {}
        
        # 生成摘要报告
        summary_file = Path(output_dir) / "summary_report.md"
        summary_content = report_generator.generate_summary_report(
            test_results, str(summary_file)
        )
        reports['summary'] = str(summary_file)
        
        # 生成性能报告
        if performance_data:
            performance_file = Path(output_dir) / "performance_report.html"
            report_generator.generate_performance_report(
                performance_data, str(performance_file)
            )
            reports['performance'] = str(performance_file)
        
        # 生成安全报告
        if security_data:
            security_file = Path(output_dir) / "security_report.html"
            report_generator.generate_security_report(
                security_data, str(security_file)
            )
            reports['security'] = str(security_file)
        
        logging.info(f"测试报告生成完成，输出目录：{output_dir}")
        return reports
        
    except Exception as e:
        logging.error(f"生成测试报告失败：{e}")
        return {}


def collect_and_export_metrics(
    session_id: str,
    test_results: List[Dict],
    environment_info: Dict = None,
    export_prometheus: bool = True
) -> Dict[str, Any]:
    """便捷的指标收集和导出函数
    
    功能：收集测试指标并可选导出到Prometheus
    参数：
        session_id: 测试会话ID
        test_results: 测试结果列表
        environment_info: 环境信息
        export_prometheus: 是否导出到Prometheus
    返回：收集的指标数据
    """
    try:
        # 收集指标
        metrics = metrics_collector.collect_test_metrics(
            session_id, test_results, environment_info
        )
        
        # 导出到Prometheus
        if export_prometheus:
            # 启动Prometheus服务器（如果尚未启动）
            metrics_exporter.start_server()
            
            # 记录测试执行
            for result in test_results:
                metrics_exporter.record_test_execution(
                    test_name=result.get('name', 'unknown'),
                    test_type=result.get('type', 'unit'),
                    status=result.get('status', 'unknown'),
                    duration=result.get('duration', 0)
                )
            
            # 更新系统指标
            metrics_exporter.update_system_metrics()
        
        return metrics
        
    except Exception as e:
        logging.error(f"收集和导出指标失败：{e}")
        return {}


def save_test_session(
    session_id: str,
    test_results: List[Dict],
    performance_data: Dict = None,
    security_data: Dict = None,
    environment_info: Dict = None,
    output_dir: str = "test_sessions"
) -> str:
    """保存完整的测试会话数据
    
    功能：保存测试会话的所有数据到文件
    参数：
        session_id: 测试会话ID
        test_results: 测试结果列表
        performance_data: 性能数据
        security_data: 安全数据
        environment_info: 环境信息
        output_dir: 输出目录
    返回：保存的文件路径
    """
    try:
        # 确保输出目录存在
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 构建会话数据
        session_data = {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'environment': environment_info or {},
            'test_results': test_results,
            'performance_data': performance_data or {},
            'security_data': security_data or {},
            'summary': {
                'total_tests': len(test_results),
                'passed_tests': sum(1 for r in test_results if r.get('status') == 'passed'),
                'failed_tests': sum(1 for r in test_results if r.get('status') == 'failed'),
                'execution_time': sum(r.get('duration', 0) for r in test_results)
            }
        }
        
        # 保存到文件
        filename = f"session_{session_id}_{int(time.time())}.json"
        filepath = Path(output_dir) / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
        
        logging.info(f"测试会话数据已保存到：{filepath}")
        return str(filepath)
        
    except Exception as e:
        logging.error(f"保存测试会话失败：{e}")
        return ""


if __name__ == "__main__":
    # 示例用法
    logging.basicConfig(level=logging.INFO)
    
    # 模拟测试结果
    test_results = [
        {
            'name': 'test_chat_completion',
            'type': 'integration',
            'status': 'passed',
            'duration': 1.5,
            'file': 'test_api.py'
        },
        {
            'name': 'test_authentication',
            'type': 'security',
            'status': 'failed',
            'duration': 0.8,
            'file': 'test_security.py'
        },
        {
            'name': 'test_performance',
            'type': 'performance',
            'status': 'passed',
            'duration': 3.2,
            'file': 'test_performance.py'
        }
    ]
    
    # 模拟性能数据
    performance_data = {
        'avg_response_time': 150.5,
        'max_response_time': 500.0,
        'throughput': 100.0,
        'error_rate': 2.5,
        'cpu_usage': 45.0,
        'memory_usage': 512.0
    }
    
    # 模拟安全数据
    security_data = {
        'vulnerabilities_found': 2,
        'critical_issues': 0,
        'high_issues': 1,
        'medium_issues': 1,
        'low_issues': 0,
        'security_score': 85.0,
        'scan_duration': 30.0
    }
    
    # 生成报告
    reports = generate_test_report(
        test_results=test_results,
        performance_data=performance_data,
        security_data=security_data,
        output_dir="example_reports"
    )
    
    print("生成的报告文件：")
    for report_type, filepath in reports.items():
        print(f"  {report_type}: {filepath}")
    
    # 收集和导出指标
    session_id = f"test_session_{int(time.time())}"
    metrics = collect_and_export_metrics(
        session_id=session_id,
        test_results=test_results,
        environment_info={'python_version': '3.9', 'os': 'Windows'},
        export_prometheus=True
    )
    
    print(f"\n收集的指标数据：")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    
    # 保存测试会话
    session_file = save_test_session(
        session_id=session_id,
        test_results=test_results,
        performance_data=performance_data,
        security_data=security_data,
        environment_info={'python_version': '3.9', 'os': 'Windows'}
    )
    
    print(f"\n测试会话已保存到：{session_file}")
    
    print("\n报告生成工具模块加载完成！")
    print(f"Prometheus指标访问地址：http://localhost:8000/metrics")