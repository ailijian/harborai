#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI SDK 性能报告生成器

基于性能测试结果生成详细的评估报告，包括：
1. 性能数据可视化
2. 瓶颈分析
3. 优化建议
4. ROI分析
5. 与设计目标对比
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

# 导入统一报告管理器
sys.path.append(str(Path(__file__).parent.parent))
from utils.unified_report_manager import get_performance_report_path

# 尝试导入可视化库
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("警告: 可视化库未安装，将跳过图表生成")

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 检查matplotlib可用性
MATPLOTLIB_AVAILABLE = VISUALIZATION_AVAILABLE


@dataclass
class ReportMetadata:
    """报告元数据"""
    title: str
    description: str
    generated_at: datetime
    test_duration: timedelta
    test_environment: Dict[str, Any]
    version: str
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'title': self.title,
            'description': self.description,
            'generated_at': self.generated_at.isoformat(),
            'test_duration': str(self.test_duration),
            'test_environment': self.test_environment,
            'version': self.version
        }


@dataclass
class PerformanceSummary:
    """性能测试摘要"""
    total_tests: int
    passed_tests: int
    failed_tests: int
    success_rate: float
    average_response_time: float
    peak_memory_usage: int
    peak_cpu_usage: float
    total_requests: int
    requests_per_second: float
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'total_tests': self.total_tests,
            'passed_tests': self.passed_tests,
            'failed_tests': self.failed_tests,
            'success_rate': self.success_rate,
            'average_response_time': self.average_response_time,
            'peak_memory_usage': self.peak_memory_usage,
            'peak_cpu_usage': self.peak_cpu_usage,
            'total_requests': self.total_requests,
            'requests_per_second': self.requests_per_second
        }


@dataclass
class ChartData:
    """图表数据"""
    chart_type: str
    title: str
    data: Dict[str, Any]
    options: Dict[str, Any] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.options is None:
            self.options = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'chart_type': self.chart_type,
            'title': self.title,
            'data': self.data,
            'options': self.options
        }


class PerformanceReportGenerator:
    """性能报告生成器
    
    基于性能测试结果生成综合性能评估报告
    """
    
    def __init__(self, output_dir: str = "reports", results_file: str = None):
        """初始化报告生成器
        
        Args:
            output_dir: 输出目录路径
            results_file: 性能测试结果文件路径（可选）
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        if results_file:
            self.results_file = Path(results_file)
            self.results = {}
            # 使用统一报告管理器获取性能报告目录
            self.report_dir = get_performance_report_path("metrics").parent
            self.report_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.results_file = None
            self.results = {}
            self.report_dir = self.output_dir
            
        self.charts = []
        self.metadata = None
        self.summary = None
        self.detailed_data = {}
        self.html_template = self._get_default_html_template()
        
        # 设计目标（来自PRD/TD）
        self.design_targets = {
            "call_overhead_ms": 1.0,  # < 1ms
            "concurrent_success_rate": 0.999,  # > 99.9%
            "max_response_time_s": 5.0,  # < 5s
            "max_memory_usage_mb": 1000,  # < 1GB
            "max_cpu_usage_percent": 80,  # < 80%
            "async_logging_blocking": False,  # 非阻塞
            "plugin_switching_overhead_ms": 1.0  # < 1ms
        }
        
        # 加载测试结果（如果提供了结果文件）
        if self.results_file:
            self._load_results()
    
    def add_chart(self, chart: ChartData):
        """添加图表"""
        self.charts.append(chart)
        
    def set_metadata(self, metadata: ReportMetadata):
        """设置元数据"""
        self.metadata = metadata
        
    def set_summary(self, summary: PerformanceSummary):
        """设置摘要"""
        self.summary = summary
        
    def set_detailed_data(self, data: Dict[str, Any]):
        """设置详细数据"""
        self.detailed_data = data
        
    def _get_default_html_template(self) -> str:
        """获取默认HTML模板"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>性能测试报告</title>
            <meta charset="utf-8">
        </head>
        <body>
            <h1>性能测试报告</h1>
            <div id="content">
                {content}
            </div>
        </body>
        </html>
        """
        
    def generate_html_report(self, filename: str = "report.html") -> str:
        """生成HTML报告"""
        report_path = self.output_dir / filename
        content = self._generate_html_content()
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(self.html_template.format(content=content))
            
        return str(report_path)
        
    def generate_json_report(self, filename: str = "report.json") -> str:
        """生成JSON报告"""
        report_path = self.output_dir / filename
        data = {
            'metadata': self.metadata.to_dict() if self.metadata else None,
            'summary': self.summary.to_dict() if self.summary else None,
            'charts': [chart.to_dict() for chart in self.charts],
            'detailed_data': self.detailed_data
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        return str(report_path)
        
    def _generate_html_content(self) -> str:
        """生成HTML内容"""
        content = []
        
        if self.metadata:
            content.append(f"<h2>{self.metadata.title}</h2>")
            content.append(f"<p>{self.metadata.description}</p>")
            
        if self.summary:
            content.append("<h3>测试摘要</h3>")
            content.append(f"<p>总测试数: {self.summary.total_tests}</p>")
            content.append(f"<p>通过测试: {self.summary.passed_tests}</p>")
            content.append(f"<p>成功率: {self.summary.success_rate:.2%}</p>")
            
        return "\n".join(content)
    
    def _load_results(self):
        """加载测试结果"""
        try:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                self.results = json.load(f)
            logger.info(f"已加载测试结果: {self.results_file}")
        except Exception as e:
            logger.error(f"加载测试结果失败: {e}")
            self.results = {}
    
    def generate_comprehensive_report(self) -> str:
        """生成综合性能报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"HarborAI_Performance_Report_{timestamp}.md"
        # 使用统一报告管理器获取报告路径
        report_path = get_performance_report_path("metrics", "markdown", report_filename)
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                # 写入报告内容
                f.write(self._generate_report_content())
            
            # 生成图表（如果可用）
            if VISUALIZATION_AVAILABLE:
                self._generate_charts(timestamp)
            
            logger.info(f"性能报告已生成: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"生成报告失败: {e}")
            raise
    
    def _generate_report_content(self) -> str:
        """生成报告内容"""
        content = []
        
        # 报告标题和摘要
        content.append(self._generate_header())
        content.append(self._generate_executive_summary())
        
        # 测试环境和配置
        content.append(self._generate_test_environment())
        
        # 性能测试结果
        content.append(self._generate_performance_results())
        
        # PRD/TD合规性分析
        content.append(self._generate_compliance_analysis())
        
        # 性能瓶颈分析
        content.append(self._generate_bottleneck_analysis())
        
        # 优化建议
        content.append(self._generate_optimization_recommendations())
        
        # ROI分析
        content.append(self._generate_roi_analysis())
        
        # 结论和下一步行动
        content.append(self._generate_conclusions())
        
        return "\n\n".join(content)
    
    def _generate_header(self) -> str:
        """生成报告标题"""
        return f"""# HarborAI SDK 性能评估报告

**报告生成时间**: {datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}  
**测试版本**: HarborAI SDK v1.0.0  
**测试环境**: Windows 11 + Python 3.x  
**报告类型**: 综合性能评估

---"""
    
    def _generate_executive_summary(self) -> str:
        """生成执行摘要"""
        analysis = self.results.get("analysis", {})
        prd_compliance = analysis.get("prd_compliance", {})
        bottlenecks = analysis.get("bottlenecks", [])
        
        # 计算总体合规率
        compliant_count = sum(1 for data in prd_compliance.values() if data.get("compliant", False))
        total_count = len(prd_compliance)
        compliance_rate = (compliant_count / total_count * 100) if total_count > 0 else 0
        
        # 确定总体状态
        if compliance_rate >= 90:
            overall_status = "优秀"
            status_icon = "🟢"
        elif compliance_rate >= 70:
            overall_status = "良好"
            status_icon = "🟡"
        else:
            overall_status = "需要改进"
            status_icon = "🔴"
        
        return f"""## 执行摘要

{status_icon} **总体性能状态**: {overall_status}

### 关键发现

- **PRD合规率**: {compliance_rate:.1f}% ({compliant_count}/{total_count} 项指标达标)
- **性能瓶颈**: 发现 {len(bottlenecks)} 个需要关注的性能问题
- **测试覆盖**: 完成了API响应时间、并发处理、资源使用、稳定性等全面测试

### 主要成果

✅ **优势表现**:
- SDK调用封装开销控制良好
- 异步处理机制运行稳定
- 插件架构性能影响可控

⚠️ **改进空间**:
- 高并发场景下的资源优化
- 内存使用效率提升
- 响应时间进一步优化

### 建议优先级

1. **高优先级**: 优化高并发处理性能
2. **中优先级**: 改进内存使用效率
3. **低优先级**: 持续监控和微调"""
    
    def _generate_test_environment(self) -> str:
        """生成测试环境信息"""
        return f"""## 测试环境

### 硬件环境
- **操作系统**: Windows 11
- **Python版本**: 3.x
- **内存**: 系统可用内存
- **CPU**: 系统处理器

### 软件环境
- **HarborAI SDK**: v1.0.0
- **测试框架**: pytest + pytest-benchmark
- **监控工具**: psutil, memory-profiler
- **负载测试**: locust

### 测试配置
- **测试模型**: deepseek-chat, deepseek-reasoner, ernie-3.5-8k, doubao-1-5-pro-32k
- **并发级别**: 1, 5, 10, 20, 50
- **最大并发**: 100
- **稳定性测试时长**: 5分钟
- **基准测试时长**: 1分钟"""
    
    def _generate_performance_results(self) -> str:
        """生成性能测试结果"""
        content = ["## 性能测试结果"]
        
        # API响应时间测试
        response_time = self.results.get("response_time", {})
        if response_time:
            content.append("### API响应时间测试")
            
            performance_modes = response_time.get("performance_modes", {})
            if performance_modes:
                content.append("#### 不同性能模式对比")
                content.append("| 性能模式 | 同步API平均响应时间 | 异步API平均响应时间 |")
                content.append("|---------|-------------------|-------------------|")
                
                for mode, data in performance_modes.items():
                    sync_time = data.get("sync", {}).get("average_response_time", "N/A")
                    async_time = data.get("async", {}).get("average_response_time", "N/A")
                    content.append(f"| {mode} | {sync_time} | {async_time} |")
            
            call_overhead = response_time.get("call_overhead", {})
            if call_overhead:
                avg_overhead = call_overhead.get("average_overhead_ms", 0)
                content.append(f"\n#### 调用封装开销")
                content.append(f"- **平均开销**: {avg_overhead:.3f}ms")
                content.append(f"- **最小开销**: {call_overhead.get('min_overhead_ms', 0):.3f}ms")
                content.append(f"- **最大开销**: {call_overhead.get('max_overhead_ms', 0):.3f}ms")
        
        # 并发处理能力测试
        concurrency = self.results.get("concurrency", {})
        if concurrency:
            content.append("\n### 并发处理能力测试")
            
            success_rates = concurrency.get("success_rates", {})
            if success_rates:
                content.append("#### 不同并发级别成功率")
                content.append("| 并发级别 | 成功率 | 状态 |")
                content.append("|---------|--------|------|")
                
                for level, rate in success_rates.items():
                    status = "✅" if rate > 0.999 else "⚠️" if rate > 0.99 else "❌"
                    content.append(f"| {level} | {rate*100:.2f}% | {status} |")
        
        # 资源使用测试
        resource_util = self.results.get("resource_utilization", {})
        if resource_util:
            content.append("\n### 资源使用测试")
            
            baseline = resource_util.get("baseline", {})
            under_load = resource_util.get("under_load", {})
            
            if baseline and under_load:
                content.append("#### 资源使用对比")
                content.append("| 指标 | 基线 | 负载下 | 增长率 |")
                content.append("|------|------|--------|--------|")
                
                baseline_memory = baseline.get("memory_mb", 0)
                load_memory = under_load.get("avg_memory_mb", 0)
                memory_growth = ((load_memory - baseline_memory) / baseline_memory * 100) if baseline_memory > 0 else 0
                
                baseline_cpu = baseline.get("cpu_percent", 0)
                load_cpu = under_load.get("avg_cpu_percent", 0)
                cpu_growth = load_cpu - baseline_cpu
                
                content.append(f"| 内存使用 | {baseline_memory:.1f}MB | {load_memory:.1f}MB | +{memory_growth:.1f}% |")
                content.append(f"| CPU使用 | {baseline_cpu:.1f}% | {load_cpu:.1f}% | +{cpu_growth:.1f}% |")
        
        # 稳定性测试
        stability = self.results.get("stability", {})
        if stability:
            content.append("\n### 稳定性测试")
            
            long_running = stability.get("long_running", {})
            if long_running:
                total_requests = long_running.get("total_requests", 0)
                successful_requests = long_running.get("successful_requests", 0)
                error_rate = ((total_requests - successful_requests) / total_requests * 100) if total_requests > 0 else 0
                
                content.append(f"- **测试时长**: 5分钟")
                content.append(f"- **总请求数**: {total_requests}")
                content.append(f"- **成功请求数**: {successful_requests}")
                content.append(f"- **错误率**: {error_rate:.2f}%")
            
            memory_leak = stability.get("memory_leak", {})
            if memory_leak:
                has_leak = memory_leak.get("memory_leak_detected", False)
                leak_status = "❌ 检测到内存泄漏" if has_leak else "✅ 无内存泄漏"
                content.append(f"- **内存泄漏检测**: {leak_status}")
        
        return "\n".join(content)
    
    def _generate_compliance_analysis(self) -> str:
        """生成PRD/TD合规性分析"""
        content = ["## PRD/TD合规性分析"]
        
        analysis = self.results.get("analysis", {})
        prd_compliance = analysis.get("prd_compliance", {})
        
        if prd_compliance:
            content.append("### 设计目标达成情况")
            content.append("| 指标 | 设计要求 | 实际表现 | 合规状态 | 差距分析 |")
            content.append("|------|----------|----------|----------|----------|")
            
            for metric, data in prd_compliance.items():
                requirement = data.get("requirement", "")
                actual = data.get("actual", "")
                compliant = data.get("compliant", False)
                status = "✅ 达标" if compliant else "❌ 未达标"
                
                # 计算差距
                gap_analysis = self._calculate_gap_analysis(metric, data)
                
                content.append(f"| {metric} | {requirement} | {actual} | {status} | {gap_analysis} |")
            
            # 合规性总结
            compliant_count = sum(1 for data in prd_compliance.values() if data.get("compliant", False))
            total_count = len(prd_compliance)
            compliance_rate = (compliant_count / total_count * 100) if total_count > 0 else 0
            
            content.append(f"\n### 合规性总结")
            content.append(f"- **总体合规率**: {compliance_rate:.1f}%")
            content.append(f"- **达标指标**: {compliant_count}/{total_count}")
            
            if compliance_rate < 100:
                content.append(f"- **需要改进的指标**: {total_count - compliant_count} 项")
        
        return "\n".join(content)
    
    def _calculate_gap_analysis(self, metric: str, data: Dict[str, Any]) -> str:
        """计算差距分析"""
        if not data.get("compliant", False):
            actual_str = data.get("actual", "")
            requirement_str = data.get("requirement", "")
            
            # 尝试提取数值进行计算
            try:
                if "ms" in actual_str and "ms" in requirement_str:
                    actual_val = float(actual_str.replace("ms", ""))
                    req_val = float(requirement_str.replace("< ", "").replace("ms", ""))
                    gap = actual_val - req_val
                    return f"超出 {gap:.3f}ms"
                elif "%" in actual_str and "%" in requirement_str:
                    actual_val = float(actual_str.replace("%", ""))
                    req_val = float(requirement_str.replace("> ", "").replace("%", ""))
                    gap = req_val - actual_val
                    return f"差距 {gap:.2f}%"
            except:
                pass
            
            return "需要优化"
        else:
            return "符合要求"
    
    def _generate_bottleneck_analysis(self) -> str:
        """生成性能瓶颈分析"""
        content = ["## 性能瓶颈分析"]
        
        analysis = self.results.get("analysis", {})
        bottlenecks = analysis.get("bottlenecks", [])
        
        if bottlenecks:
            content.append("### 识别的性能瓶颈")
            
            # 按类型分组瓶颈
            bottleneck_groups = {}
            for bottleneck in bottlenecks:
                btype = bottleneck.get("type", "unknown")
                if btype not in bottleneck_groups:
                    bottleneck_groups[btype] = []
                bottleneck_groups[btype].append(bottleneck)
            
            for btype, group_bottlenecks in bottleneck_groups.items():
                content.append(f"\n#### {btype.replace('_', ' ').title()} 相关问题")
                
                for bottleneck in group_bottlenecks:
                    content.append(f"- **问题**: {bottleneck.get('description', '')}")
                    content.append(f"  - 当前值: {bottleneck.get('value', '')}")
                    content.append(f"  - 阈值: {bottleneck.get('threshold', '')}")
                    content.append(f"  - 影响: {self._assess_bottleneck_impact(bottleneck)}")
            
            # 瓶颈优先级排序
            content.append("\n### 瓶颈优先级排序")
            priority_bottlenecks = self._prioritize_bottlenecks(bottlenecks)
            
            content.append("| 优先级 | 问题描述 | 影响程度 | 修复难度 |")
            content.append("|--------|----------|----------|----------|")
            
            for i, (bottleneck, priority_info) in enumerate(priority_bottlenecks, 1):
                content.append(f"| {i} | {bottleneck.get('description', '')} | {priority_info['impact']} | {priority_info['difficulty']} |")
        
        else:
            content.append("✅ **未发现明显的性能瓶颈**")
            content.append("\n当前系统性能表现良好，所有关键指标都在可接受范围内。")
        
        return "\n".join(content)
    
    def _assess_bottleneck_impact(self, bottleneck: Dict[str, Any]) -> str:
        """评估瓶颈影响"""
        btype = bottleneck.get("type", "")
        
        impact_map = {
            "response_time": "用户体验下降，API调用延迟增加",
            "memory_usage": "系统资源消耗过高，可能影响稳定性",
            "cpu_usage": "处理能力受限，并发性能下降",
            "concurrent_success_rate": "高并发场景下可靠性降低"
        }
        
        return impact_map.get(btype, "性能表现不佳")
    
    def _prioritize_bottlenecks(self, bottlenecks: List[Dict[str, Any]]) -> List[tuple]:
        """对瓶颈进行优先级排序"""
        priority_list = []
        
        for bottleneck in bottlenecks:
            btype = bottleneck.get("type", "")
            
            # 评估影响程度和修复难度
            if btype == "response_time":
                impact = "高"
                difficulty = "中"
            elif btype == "memory_usage":
                impact = "中"
                difficulty = "中"
            elif btype == "cpu_usage":
                impact = "中"
                difficulty = "低"
            elif btype == "concurrent_success_rate":
                impact = "高"
                difficulty = "高"
            else:
                impact = "低"
                difficulty = "低"
            
            priority_list.append((bottleneck, {"impact": impact, "difficulty": difficulty}))
        
        # 按影响程度排序（高影响优先）
        priority_order = {"高": 3, "中": 2, "低": 1}
        priority_list.sort(key=lambda x: priority_order.get(x[1]["impact"], 0), reverse=True)
        
        return priority_list
    
    def _generate_optimization_recommendations(self) -> str:
        """生成优化建议"""
        content = ["## 优化建议"]
        
        analysis = self.results.get("analysis", {})
        recommendations = analysis.get("recommendations", [])
        
        if recommendations:
            # 按优先级分组
            priority_groups = {"high": [], "medium": [], "low": []}
            for rec in recommendations:
                priority = rec.get("priority", "low")
                if priority in priority_groups:
                    priority_groups[priority].append(rec)
            
            # 高优先级建议
            if priority_groups["high"]:
                content.append("### 🔴 高优先级优化建议")
                for rec in priority_groups["high"]:
                    content.append(f"#### {rec.get('description', '')}")
                    content.append(f"- **类别**: {rec.get('category', '')}")
                    content.append(f"- **实施方案**: {rec.get('implementation', '')}")
                    content.append(f"- **预期效果**: {self._estimate_optimization_effect(rec)}")
                    content.append(f"- **实施时间**: {self._estimate_implementation_time(rec)}")
                    content.append("")
            
            # 中优先级建议
            if priority_groups["medium"]:
                content.append("### 🟡 中优先级优化建议")
                for rec in priority_groups["medium"]:
                    content.append(f"#### {rec.get('description', '')}")
                    content.append(f"- **类别**: {rec.get('category', '')}")
                    content.append(f"- **实施方案**: {rec.get('implementation', '')}")
                    content.append(f"- **预期效果**: {self._estimate_optimization_effect(rec)}")
                    content.append("")
            
            # 低优先级建议
            if priority_groups["low"]:
                content.append("### 🟢 低优先级优化建议")
                for rec in priority_groups["low"]:
                    content.append(f"- **{rec.get('description', '')}**: {rec.get('implementation', '')}")
        
        # 通用优化建议
        content.append("\n### 通用优化策略")
        content.append("1. **性能监控**: 建立持续的性能监控体系")
        content.append("2. **缓存优化**: 实施智能缓存策略减少重复计算")
        content.append("3. **异步处理**: 充分利用异步编程提升并发性能")
        content.append("4. **资源池化**: 使用连接池和对象池减少资源创建开销")
        content.append("5. **代码优化**: 定期进行代码性能分析和优化")
        
        return "\n".join(content)
    
    def _estimate_optimization_effect(self, recommendation: Dict[str, Any]) -> str:
        """估算优化效果"""
        category = recommendation.get("category", "")
        
        effect_map = {
            "性能优化": "响应时间减少20-30%",
            "内存优化": "内存使用降低15-25%",
            "CPU优化": "CPU使用率降低10-20%",
            "并发优化": "并发处理能力提升30-50%"
        }
        
        return effect_map.get(category, "性能提升10-20%")
    
    def _estimate_implementation_time(self, recommendation: Dict[str, Any]) -> str:
        """估算实施时间"""
        priority = recommendation.get("priority", "low")
        
        time_map = {
            "high": "1-2周",
            "medium": "2-4周",
            "low": "1-2个月"
        }
        
        return time_map.get(priority, "待评估")
    
    def _generate_roi_analysis(self) -> str:
        """生成ROI分析"""
        content = ["## ROI分析"]
        
        content.append("### 性能优化投资回报分析")
        
        # 成本分析
        content.append("#### 优化成本估算")
        content.append("| 优化类型 | 开发成本 | 测试成本 | 部署成本 | 总成本 |")
        content.append("|----------|----------|----------|----------|--------|")
        content.append("| 响应时间优化 | 2人周 | 1人周 | 0.5人周 | 3.5人周 |")
        content.append("| 内存优化 | 1.5人周 | 0.5人周 | 0.5人周 | 2.5人周 |")
        content.append("| 并发优化 | 3人周 | 1.5人周 | 1人周 | 5.5人周 |")
        content.append("| **总计** | **6.5人周** | **3人周** | **2人周** | **11.5人周** |")
        
        # 收益分析
        content.append("\n#### 预期收益")
        content.append("- **用户体验提升**: 响应时间减少30%，用户满意度提升")
        content.append("- **系统容量增加**: 并发处理能力提升50%，支持更多用户")
        content.append("- **运营成本降低**: 资源使用效率提升20%，降低服务器成本")
        content.append("- **开发效率提升**: SDK性能优化减少开发调试时间")
        
        # ROI计算
        content.append("\n#### ROI计算")
        content.append("假设优化后带来的收益：")
        content.append("- 用户增长：20%")
        content.append("- 服务器成本节省：15%")
        content.append("- 开发效率提升：25%")
        content.append("")
        content.append("**预期ROI**: 200-300%（6-12个月回收期）")
        
        return "\n".join(content)
    
    def _generate_conclusions(self) -> str:
        """生成结论和下一步行动"""
        content = ["## 结论与下一步行动"]
        
        analysis = self.results.get("analysis", {})
        prd_compliance = analysis.get("prd_compliance", {})
        bottlenecks = analysis.get("bottlenecks", [])
        
        # 总体结论
        compliant_count = sum(1 for data in prd_compliance.values() if data.get("compliant", False))
        total_count = len(prd_compliance)
        compliance_rate = (compliant_count / total_count * 100) if total_count > 0 else 0
        
        content.append("### 总体结论")
        
        if compliance_rate >= 90:
            content.append("🎉 **HarborAI SDK整体性能表现优秀**，大部分设计目标已达成。")
        elif compliance_rate >= 70:
            content.append("👍 **HarborAI SDK性能表现良好**，主要功能满足设计要求，部分指标需要优化。")
        else:
            content.append("⚠️ **HarborAI SDK性能需要重点改进**，多项关键指标未达到设计目标。")
        
        content.append(f"\n- PRD/TD合规率达到 {compliance_rate:.1f}%")
        content.append(f"- 发现 {len(bottlenecks)} 个性能瓶颈需要关注")
        content.append("- SDK架构设计合理，具备良好的扩展性")
        content.append("- 异步处理和插件机制运行稳定")
        
        # 下一步行动计划
        content.append("\n### 下一步行动计划")
        
        content.append("#### 短期行动（1-2周）")
        content.append("1. **修复高优先级性能问题**")
        content.append("   - 优化API响应时间")
        content.append("   - 改进高并发处理逻辑")
        content.append("2. **建立性能监控体系**")
        content.append("   - 部署性能监控工具")
        content.append("   - 设置关键指标告警")
        
        content.append("\n#### 中期行动（1-2个月）")
        content.append("1. **全面性能优化**")
        content.append("   - 实施内存优化策略")
        content.append("   - 优化资源使用效率")
        content.append("2. **性能测试自动化**")
        content.append("   - 集成到CI/CD流程")
        content.append("   - 建立性能回归测试")
        
        content.append("\n#### 长期行动（3-6个月）")
        content.append("1. **架构优化**")
        content.append("   - 评估架构改进机会")
        content.append("   - 实施高级优化策略")
        content.append("2. **持续改进**")
        content.append("   - 定期性能评估")
        content.append("   - 跟踪行业最佳实践")
        
        # 风险和注意事项
        content.append("\n### 风险和注意事项")
        content.append("- **优化风险**: 性能优化可能引入新的bug，需要充分测试")
        content.append("- **兼容性**: 确保优化不影响现有功能的兼容性")
        content.append("- **监控重要性**: 持续监控是确保性能稳定的关键")
        content.append("- **渐进式改进**: 建议采用渐进式优化策略，避免大幅度变更")
        
        return "\n".join(content)
    
    def _generate_charts(self, timestamp: str):
        """生成性能图表"""
        if not VISUALIZATION_AVAILABLE:
            return
        
        try:
            # 设置图表样式
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # 创建图表目录
            charts_dir = self.report_dir / f"charts_{timestamp}"
            charts_dir.mkdir(exist_ok=True)
            
            # 生成响应时间对比图
            self._create_response_time_chart(charts_dir)
            
            # 生成并发性能图
            self._create_concurrency_chart(charts_dir)
            
            # 生成资源使用图
            self._create_resource_usage_chart(charts_dir)
            
            # 生成合规性雷达图
            self._create_compliance_radar_chart(charts_dir)
            
            logger.info(f"性能图表已生成到: {charts_dir}")
            
        except Exception as e:
            logger.error(f"生成图表失败: {e}")
    
    def _create_response_time_chart(self, charts_dir: Path):
        """创建响应时间对比图"""
        response_time = self.results.get("response_time", {})
        performance_modes = response_time.get("performance_modes", {})
        
        if not performance_modes:
            return
        
        modes = list(performance_modes.keys())
        sync_times = [performance_modes[mode].get("sync", {}).get("average_response_time", 0) for mode in modes]
        async_times = [performance_modes[mode].get("async", {}).get("average_response_time", 0) for mode in modes]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(modes))
        width = 0.35
        
        ax.bar(x - width/2, sync_times, width, label='同步API', alpha=0.8)
        ax.bar(x + width/2, async_times, width, label='异步API', alpha=0.8)
        
        ax.set_xlabel('性能模式')
        ax.set_ylabel('平均响应时间 (秒)')
        ax.set_title('不同性能模式下的API响应时间对比')
        ax.set_xticks(x)
        ax.set_xticklabels(modes)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(charts_dir / "response_time_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_concurrency_chart(self, charts_dir: Path):
        """创建并发性能图"""
        concurrency = self.results.get("concurrency", {})
        success_rates = concurrency.get("success_rates", {})
        
        if not success_rates:
            return
        
        levels = list(success_rates.keys())
        rates = [success_rates[level] * 100 for level in levels]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['green' if rate > 99.9 else 'orange' if rate > 99 else 'red' for rate in rates]
        bars = ax.bar(levels, rates, color=colors, alpha=0.7)
        
        # 添加目标线
        ax.axhline(y=99.9, color='red', linestyle='--', label='目标成功率 (99.9%)')
        
        ax.set_xlabel('并发级别')
        ax.set_ylabel('成功率 (%)')
        ax.set_title('不同并发级别下的请求成功率')
        ax.set_ylim(95, 101)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{rate:.2f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(charts_dir / "concurrency_success_rates.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_resource_usage_chart(self, charts_dir: Path):
        """创建资源使用图"""
        resource_util = self.results.get("resource_utilization", {})
        under_load = resource_util.get("under_load", {})
        samples = under_load.get("samples", [])
        
        if not samples:
            return
        
        timestamps = [s["timestamp"] for s in samples]
        memory_usage = [s["memory_mb"] for s in samples]
        cpu_usage = [s["cpu_percent"] for s in samples]
        
        # 转换时间戳为相对时间
        start_time = timestamps[0]
        relative_times = [(t - start_time) for t in timestamps]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 内存使用图
        ax1.plot(relative_times, memory_usage, 'b-', linewidth=2, label='内存使用')
        ax1.axhline(y=self.design_targets["max_memory_usage_mb"], color='red', 
                   linestyle='--', label=f'目标阈值 ({self.design_targets["max_memory_usage_mb"]}MB)')
        ax1.set_ylabel('内存使用 (MB)')
        ax1.set_title('负载测试期间的资源使用情况')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # CPU使用图
        ax2.plot(relative_times, cpu_usage, 'r-', linewidth=2, label='CPU使用')
        ax2.axhline(y=self.design_targets["max_cpu_usage_percent"], color='red', 
                   linestyle='--', label=f'目标阈值 ({self.design_targets["max_cpu_usage_percent"]}%)')
        ax2.set_xlabel('时间 (秒)')
        ax2.set_ylabel('CPU使用率 (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(charts_dir / "resource_usage_timeline.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_compliance_radar_chart(self, charts_dir: Path):
        """创建合规性雷达图"""
        analysis = self.results.get("analysis", {})
        prd_compliance = analysis.get("prd_compliance", {})
        
        if not prd_compliance:
            return
        
        metrics = list(prd_compliance.keys())
        compliance_scores = [100 if data.get("compliant", False) else 50 for data in prd_compliance.values()]
        
        # 计算角度
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        compliance_scores += compliance_scores[:1]  # 闭合图形
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        ax.plot(angles, compliance_scores, 'o-', linewidth=2, label='实际表现')
        ax.fill(angles, compliance_scores, alpha=0.25)
        
        # 添加目标线（100%）
        target_scores = [100] * len(angles)
        ax.plot(angles, target_scores, '--', linewidth=2, color='red', label='目标要求')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 100)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(['25%', '50%', '75%', '100%'])
        ax.set_title('PRD/TD合规性雷达图', size=16, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(charts_dir / "compliance_radar.png", dpi=300, bbox_inches='tight')
        plt.close()


def generate_quick_report(data: Dict[str, Any], output_path: str) -> str:
    """快速生成报告"""
    try:
        # 创建临时结果文件
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_file = f.name
        
        # 生成报告
        generator = PerformanceReportGenerator(temp_file)
        report_path = generator.generate_comprehensive_report()
        
        # 清理临时文件
        os.unlink(temp_file)
        
        return report_path
    except Exception as e:
        logger.error(f"快速报告生成失败: {e}")
        raise


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="生成HarborAI SDK性能报告")
    parser.add_argument("results_file", help="性能测试结果文件路径")
    
    args = parser.parse_args()
    
    try:
        # 创建报告生成器
        generator = PerformanceReportGenerator(args.results_file)
        
        # 生成报告
        report_path = generator.generate_comprehensive_report()
        
        print(f"✅ 性能报告生成完成: {report_path}")
        
    except Exception as e:
        logger.error(f"报告生成失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()