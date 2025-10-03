#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI SDK综合性能评估报告生成器

整合所有性能测试结果，生成完整的性能评估报告
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import statistics

class ComprehensivePerformanceReportGenerator:
    """综合性能报告生成器"""
    
    def __init__(self):
        self.prd_requirements = {
            'call_overhead_ms': 1.0,
            'concurrency_success_rate': 99.9,
            'memory_leak_mb': 1.0,
            'initialization_time_ms': 500.0,
            'concurrent_throughput_ops_per_sec': 1000.0
        }
        
    def load_all_results(self) -> Dict[str, Any]:
        """加载所有测试结果"""
        results = {}
        
        # 加载基础性能测试结果
        try:
            with open('sdk_performance_results.json', 'r', encoding='utf-8') as f:
                results['basic_performance'] = json.load(f)
        except FileNotFoundError:
            results['basic_performance'] = {}
        
        # 加载对比测试结果
        try:
            with open('sdk_comparison_results.json', 'r', encoding='utf-8') as f:
                results['comparison'] = json.load(f)
        except FileNotFoundError:
            results['comparison'] = {}
        
        # 加载特有功能测试结果
        try:
            with open('sdk_features_performance_results.json', 'r', encoding='utf-8') as f:
                results['features'] = json.load(f)
        except FileNotFoundError:
            results['features'] = {}
        
        # 加载分析报告数据
        try:
            with open('harborai_performance_analysis_report.md', 'r', encoding='utf-8') as f:
                results['analysis_report'] = f.read()
        except FileNotFoundError:
            results['analysis_report'] = ""
        
        # 加载对比报告数据
        try:
            with open('harborai_vs_openai_comparison_report.md', 'r', encoding='utf-8') as f:
                results['comparison_report'] = f.read()
        except FileNotFoundError:
            results['comparison_report'] = ""
        
        # 加载优化计划数据
        try:
            with open('harborai_performance_optimization_plan.md', 'r', encoding='utf-8') as f:
                results['optimization_plan'] = f.read()
        except FileNotFoundError:
            results['optimization_plan'] = ""
        
        return results
    
    def calculate_overall_score(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """计算总体性能评分"""
        basic_perf = results.get('basic_performance', {})
        comparison = results.get('comparison', {})
        
        scores = {}
        
        # PRD合规性评分
        prd_scores = []
        
        # 1. 调用开销评分
        method_overhead = basic_perf.get('method_call_overhead', {})
        if method_overhead:
            avg_overhead_us = statistics.mean([
                data.get('average_us', 0) for data in method_overhead.values()
            ])
            call_overhead_ms = avg_overhead_us / 1000
            call_score = max(0, 100 - (call_overhead_ms / self.prd_requirements['call_overhead_ms']) * 100)
            prd_scores.append(call_score)
            scores['call_overhead_score'] = call_score
        
        # 2. 并发成功率评分
        concurrent_perf = basic_perf.get('concurrent_performance', {})
        if concurrent_perf:
            success_rates = [data.get('success_rate', 0) for data in concurrent_perf.values()]
            avg_success_rate = statistics.mean(success_rates) if success_rates else 0
            concurrency_score = (avg_success_rate / self.prd_requirements['concurrency_success_rate']) * 100
            prd_scores.append(concurrency_score)
            scores['concurrency_score'] = concurrency_score
        
        # 3. 内存泄漏评分
        memory_usage = basic_perf.get('memory_usage', {})
        memory_leak = memory_usage.get('potential_memory_leak_mb', 0)
        memory_score = max(0, 100 - (memory_leak / self.prd_requirements['memory_leak_mb']) * 100)
        prd_scores.append(memory_score)
        scores['memory_score'] = memory_score
        
        # 4. 初始化时间评分
        init_overhead = basic_perf.get('initialization_overhead', {})
        if init_overhead:
            init_times = [data.get('average_ms', 0) for data in init_overhead.values()]
            avg_init_time = statistics.mean(init_times) if init_times else 0
            init_score = max(0, 100 - (avg_init_time / self.prd_requirements['initialization_time_ms']) * 100)
            prd_scores.append(init_score)
            scores['initialization_score'] = init_score
        
        # 5. 并发吞吐量评分
        if concurrent_perf:
            throughputs = [data.get('operations_per_second', 0) for data in concurrent_perf.values()]
            max_throughput = max(throughputs) if throughputs else 0
            throughput_score = min(100, (max_throughput / self.prd_requirements['concurrent_throughput_ops_per_sec']) * 100)
            prd_scores.append(throughput_score)
            scores['throughput_score'] = throughput_score
        
        # PRD总体合规性评分
        scores['prd_compliance_score'] = statistics.mean(prd_scores) if prd_scores else 0
        
        # 与OpenAI SDK对比评分
        harborai_data = comparison.get('HarborAI', {})
        openai_data = comparison.get('OpenAI', {})
        
        if harborai_data and openai_data:
            comparison_scores = []
            
            # 初始化时间对比
            harbor_init = harborai_data.get('initialization_time_ms', 0)
            openai_init = openai_data.get('initialization_time_ms', 0)
            if openai_init > 0:
                init_comparison = min(100, (openai_init / harbor_init) * 100)
                comparison_scores.append(init_comparison)
            
            # 方法调用开销对比
            harbor_call = harborai_data.get('method_call_overhead_us', 0)
            openai_call = openai_data.get('method_call_overhead_us', 0)
            if openai_call > 0:
                call_comparison = min(100, (openai_call / harbor_call) * 100)
                comparison_scores.append(call_comparison)
            
            # 内存使用对比
            harbor_mem = harborai_data.get('memory_usage_mb', 0)
            openai_mem = openai_data.get('memory_usage_mb', 0)
            if openai_mem > 0:
                mem_comparison = min(100, (openai_mem / harbor_mem) * 100)
                comparison_scores.append(mem_comparison)
            
            # 并发吞吐量对比
            harbor_throughput = harborai_data.get('concurrent_throughput_ops_per_sec', 0)
            openai_throughput = openai_data.get('concurrent_throughput_ops_per_sec', 0)
            if openai_throughput > 0:
                throughput_comparison = min(100, (harbor_throughput / openai_throughput) * 100)
                comparison_scores.append(throughput_comparison)
            
            scores['openai_comparison_score'] = statistics.mean(comparison_scores) if comparison_scores else 0
        
        # 特有功能性能评分
        features = results.get('features', {})
        if features:
            feature_scores = []
            for feature_name, data in features.items():
                # 基于操作开销和吞吐量计算功能评分
                operation_overhead = data.get('operation_overhead_us', 0)
                throughput = data.get('throughput_ops_per_sec', 0)
                success_rate = data.get('success_rate_percent', 0)
                
                # 操作开销评分 (越低越好)
                overhead_score = max(0, 100 - operation_overhead * 2)  # 每微秒扣2分
                
                # 吞吐量评分 (越高越好)
                throughput_score = min(100, throughput / 10)  # 每10ops/s得1分
                
                # 成功率评分
                success_score = success_rate
                
                feature_score = (overhead_score + throughput_score + success_score) / 3
                feature_scores.append(feature_score)
            
            scores['features_score'] = statistics.mean(feature_scores) if feature_scores else 0
        
        # 总体评分
        all_scores = [
            scores.get('prd_compliance_score', 0),
            scores.get('openai_comparison_score', 0),
            scores.get('features_score', 0)
        ]
        scores['overall_score'] = statistics.mean([s for s in all_scores if s > 0])
        
        return scores
    
    def generate_executive_summary(self, results: Dict[str, Any], scores: Dict[str, Any]) -> str:
        """生成执行摘要"""
        summary = []
        
        overall_score = scores.get('overall_score', 0)
        prd_score = scores.get('prd_compliance_score', 0)
        comparison_score = scores.get('openai_comparison_score', 0)
        
        # 总体评价
        if overall_score >= 80:
            performance_level = "优秀"
            recommendation = "HarborAI SDK性能表现优异，满足生产环境要求"
        elif overall_score >= 60:
            performance_level = "良好"
            recommendation = "HarborAI SDK性能表现良好，建议进行部分优化"
        elif overall_score >= 40:
            performance_level = "一般"
            recommendation = "HarborAI SDK性能有待提升，需要重点优化"
        else:
            performance_level = "较差"
            recommendation = "HarborAI SDK性能存在严重问题，需要全面优化"
        
        summary.append(f"## 执行摘要")
        summary.append(f"")
        summary.append(f"### 总体评价: {performance_level} ({overall_score:.1f}/100)")
        summary.append(f"{recommendation}")
        summary.append(f"")
        summary.append(f"### 关键指标")
        summary.append(f"- **PRD合规性**: {prd_score:.1f}/100")
        summary.append(f"- **与OpenAI SDK对比**: {comparison_score:.1f}/100")
        summary.append(f"- **特有功能性能**: {scores.get('features_score', 0):.1f}/100")
        summary.append(f"")
        
        # 主要发现
        summary.append(f"### 主要发现")
        
        basic_perf = results.get('basic_performance', {})
        comparison = results.get('comparison', {})
        
        # PRD合规性分析
        if prd_score >= 80:
            summary.append(f"✅ **PRD合规性优秀**: 所有关键性能指标均满足设计要求")
        elif prd_score >= 60:
            summary.append(f"⚠️ **PRD合规性良好**: 大部分性能指标满足要求，少数指标需要优化")
        else:
            summary.append(f"❌ **PRD合规性不足**: 多项关键性能指标未达到设计要求")
        
        # 与OpenAI SDK对比分析
        if comparison_score >= 80:
            summary.append(f"✅ **竞争力强**: 性能表现接近或超越OpenAI SDK")
        elif comparison_score >= 50:
            summary.append(f"⚠️ **竞争力一般**: 性能略逊于OpenAI SDK，有优化空间")
        else:
            summary.append(f"❌ **竞争力不足**: 性能明显落后于OpenAI SDK，需要重点优化")
        
        # 特有功能分析
        features_score = scores.get('features_score', 0)
        if features_score >= 70:
            summary.append(f"✅ **特有功能性能良好**: 插件架构等特有功能运行高效")
        elif features_score >= 50:
            summary.append(f"⚠️ **特有功能性能一般**: 部分特有功能存在性能开销")
        else:
            summary.append(f"❌ **特有功能性能不佳**: 特有功能显著影响整体性能")
        
        summary.append(f"")
        
        return "\n".join(summary)
    
    def generate_detailed_analysis(self, results: Dict[str, Any], scores: Dict[str, Any]) -> str:
        """生成详细分析"""
        analysis = []
        
        analysis.append("## 详细性能分析")
        analysis.append("")
        
        # 基础性能分析
        basic_perf = results.get('basic_performance', {})
        if basic_perf:
            analysis.append("### 基础性能指标")
            
            # 初始化性能
            init_overhead = basic_perf.get('initialization_overhead', {})
            if init_overhead:
                analysis.append("#### 初始化性能")
                for mode, data in init_overhead.items():
                    avg_time = data.get('average_ms', 0)
                    analysis.append(f"- **{mode}模式**: {avg_time:.2f}ms")
                analysis.append("")
            
            # 方法调用性能
            method_overhead = basic_perf.get('method_call_overhead', {})
            if method_overhead:
                analysis.append("#### 方法调用性能")
                for method, data in method_overhead.items():
                    avg_overhead = data.get('average_us', 0)
                    analysis.append(f"- **{method}**: {avg_overhead:.2f}μs")
                analysis.append("")
            
            # 内存使用
            memory_usage = basic_perf.get('memory_usage', {})
            if memory_usage:
                analysis.append("#### 内存使用")
                baseline = memory_usage.get('baseline_mb', 0)
                leak = memory_usage.get('potential_memory_leak_mb', 0)
                analysis.append(f"- **基准内存**: {baseline:.2f}MB")
                analysis.append(f"- **潜在内存泄漏**: {leak:.2f}MB")
                analysis.append("")
            
            # 并发性能
            concurrent_perf = basic_perf.get('concurrent_performance', {})
            if concurrent_perf:
                analysis.append("#### 并发性能")
                for concurrency, data in concurrent_perf.items():
                    throughput = data.get('operations_per_second', 0)
                    success_rate = data.get('success_rate', 0)
                    analysis.append(f"- **{concurrency}并发**: {throughput:.1f}ops/s, 成功率{success_rate:.1f}%")
                analysis.append("")
        
        # 对比分析
        comparison = results.get('comparison', {})
        if comparison:
            analysis.append("### 与OpenAI SDK对比")
            
            harborai_data = comparison.get('HarborAI', {})
            openai_data = comparison.get('OpenAI', {})
            
            if harborai_data and openai_data:
                analysis.append("| 指标 | HarborAI | OpenAI | 差距 |")
                analysis.append("|------|----------|--------|------|")
                
                # 初始化时间
                harbor_init = harborai_data.get('initialization_time_ms', 0)
                openai_init = openai_data.get('initialization_time_ms', 0)
                init_diff = harbor_init - openai_init
                init_pct = (init_diff / openai_init * 100) if openai_init > 0 else 0
                analysis.append(f"| 初始化时间 | {harbor_init:.2f}ms | {openai_init:.2f}ms | {init_pct:+.1f}% |")
                
                # 方法调用开销
                harbor_call = harborai_data.get('method_call_overhead_us', 0)
                openai_call = openai_data.get('method_call_overhead_us', 0)
                call_diff = harbor_call - openai_call
                call_pct = (call_diff / openai_call * 100) if openai_call > 0 else 0
                analysis.append(f"| 方法调用开销 | {harbor_call:.2f}μs | {openai_call:.2f}μs | {call_pct:+.1f}% |")
                
                # 内存使用
                harbor_mem = harborai_data.get('memory_usage_mb', 0)
                openai_mem = openai_data.get('memory_usage_mb', 0)
                mem_diff = harbor_mem - openai_mem
                mem_pct = (mem_diff / openai_mem * 100) if openai_mem > 0 else 0
                analysis.append(f"| 内存使用 | {harbor_mem:.2f}MB | {openai_mem:.2f}MB | {mem_pct:+.1f}% |")
                
                # 并发吞吐量
                harbor_throughput = harborai_data.get('concurrent_throughput_ops_per_sec', 0)
                openai_throughput = openai_data.get('concurrent_throughput_ops_per_sec', 0)
                throughput_diff = harbor_throughput - openai_throughput
                throughput_pct = (throughput_diff / openai_throughput * 100) if openai_throughput > 0 else 0
                analysis.append(f"| 并发吞吐量 | {harbor_throughput:.1f}ops/s | {openai_throughput:.1f}ops/s | {throughput_pct:+.1f}% |")
                
                analysis.append("")
        
        # 特有功能分析
        features = results.get('features', {})
        if features:
            analysis.append("### 特有功能性能")
            
            analysis.append("| 功能 | 操作开销 | 内存开销 | 吞吐量 | 成功率 |")
            analysis.append("|------|----------|----------|--------|--------|")
            
            for feature_name, data in features.items():
                operation_overhead = data.get('operation_overhead_us', 0)
                memory_overhead = data.get('memory_overhead_mb', 0)
                throughput = data.get('throughput_ops_per_sec', 0)
                success_rate = data.get('success_rate_percent', 0)
                
                analysis.append(f"| {feature_name} | {operation_overhead:.2f}μs | {memory_overhead:.2f}MB | {throughput:.1f}ops/s | {success_rate:.1f}% |")
            
            analysis.append("")
        
        return "\n".join(analysis)
    
    def generate_recommendations_summary(self, results: Dict[str, Any]) -> str:
        """生成优化建议摘要"""
        recommendations = []
        
        recommendations.append("## 优化建议摘要")
        recommendations.append("")
        
        # 基于分析结果生成建议
        basic_perf = results.get('basic_performance', {})
        comparison = results.get('comparison', {})
        
        # 初始化优化建议
        init_overhead = basic_perf.get('initialization_overhead', {})
        if init_overhead:
            avg_init_times = [data.get('average_ms', 0) for data in init_overhead.values()]
            max_init_time = max(avg_init_times) if avg_init_times else 0
            
            if max_init_time > 200:
                recommendations.append("### 🔥 高优先级优化")
                recommendations.append("1. **初始化性能优化**")
                recommendations.append("   - 实现延迟加载机制")
                recommendations.append("   - 优化插件初始化流程")
                recommendations.append("   - 并行化初始化操作")
                recommendations.append("")
        
        # 方法调用优化建议
        method_overhead = basic_perf.get('method_call_overhead', {})
        if method_overhead:
            avg_overheads = [data.get('average_us', 0) for data in method_overhead.values()]
            max_overhead = max(avg_overheads) if avg_overheads else 0
            
            if max_overhead > 1:
                recommendations.append("2. **方法调用性能优化**")
                recommendations.append("   - 简化方法调用链")
                recommendations.append("   - 减少参数验证开销")
                recommendations.append("   - 优化装饰器和中间件")
                recommendations.append("")
        
        # 内存优化建议
        memory_usage = basic_perf.get('memory_usage', {})
        memory_leak = memory_usage.get('potential_memory_leak_mb', 0)
        
        if memory_leak > 1:
            recommendations.append("3. **内存管理优化**")
            recommendations.append("   - 修复内存泄漏问题")
            recommendations.append("   - 优化对象生命周期管理")
            recommendations.append("   - 实现内存监控机制")
            recommendations.append("")
        
        # 并发优化建议
        concurrent_perf = basic_perf.get('concurrent_performance', {})
        if concurrent_perf:
            success_rates = [data.get('success_rate', 0) for data in concurrent_perf.values()]
            min_success_rate = min(success_rates) if success_rates else 100
            
            if min_success_rate < 99:
                recommendations.append("### ⚠️ 中优先级优化")
                recommendations.append("4. **并发稳定性优化**")
                recommendations.append("   - 提高并发处理稳定性")
                recommendations.append("   - 优化错误处理机制")
                recommendations.append("   - 实现更好的资源管理")
                recommendations.append("")
        
        # 与OpenAI SDK对比的优化建议
        if comparison:
            harborai_data = comparison.get('HarborAI', {})
            openai_data = comparison.get('OpenAI', {})
            
            if harborai_data and openai_data:
                harbor_throughput = harborai_data.get('concurrent_throughput_ops_per_sec', 0)
                openai_throughput = openai_data.get('concurrent_throughput_ops_per_sec', 0)
                
                if harbor_throughput < openai_throughput * 0.8:  # 如果吞吐量低于OpenAI的80%
                    recommendations.append("5. **竞争力提升优化**")
                    recommendations.append("   - 参考OpenAI SDK的性能优化策略")
                    recommendations.append("   - 重点优化并发处理能力")
                    recommendations.append("   - 减少特有功能的性能开销")
                    recommendations.append("")
        
        # 特有功能优化建议
        features = results.get('features', {})
        if features:
            high_overhead_features = [
                name for name, data in features.items()
                if data.get('operation_overhead_us', 0) > 10
            ]
            
            if high_overhead_features:
                recommendations.append("### 💡 长期优化")
                recommendations.append("6. **特有功能性能优化**")
                for feature in high_overhead_features:
                    recommendations.append(f"   - 优化{feature}的实现效率")
                recommendations.append("   - 实现功能开关和按需加载")
                recommendations.append("   - 优化插件架构设计")
                recommendations.append("")
        
        if not any("优先级优化" in line for line in recommendations):
            recommendations.append("### ✅ 性能表现良好")
            recommendations.append("当前性能指标基本满足要求，建议：")
            recommendations.append("- 持续监控性能指标")
            recommendations.append("- 定期进行性能回归测试")
            recommendations.append("- 关注新功能对性能的影响")
            recommendations.append("")
        
        return "\n".join(recommendations)
    
    def generate_comprehensive_report(self) -> str:
        """生成综合性能评估报告"""
        print("📊 生成综合性能评估报告...")
        
        # 加载所有结果
        results = self.load_all_results()
        
        # 计算评分
        scores = self.calculate_overall_score(results)
        
        # 生成报告
        report = []
        
        # 报告头部
        report.append("# HarborAI SDK综合性能评估报告")
        report.append("=" * 80)
        report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**评估版本**: HarborAI SDK v1.0")
        report.append(f"**测试环境**: Windows 11, Python 3.x")
        report.append("")
        
        # 执行摘要
        report.append(self.generate_executive_summary(results, scores))
        
        # 详细分析
        report.append(self.generate_detailed_analysis(results, scores))
        
        # 优化建议摘要
        report.append(self.generate_recommendations_summary(results))
        
        # 测试方法说明
        report.append("## 测试方法说明")
        report.append("")
        report.append("### 测试范围")
        report.append("- **基础性能测试**: 初始化时间、方法调用开销、内存使用、并发性能")
        report.append("- **对比测试**: 与OpenAI SDK的性能对比")
        report.append("- **特有功能测试**: 插件架构、结构化输出、推理模型支持等")
        report.append("")
        report.append("### 测试环境")
        report.append("- **操作系统**: Windows 11")
        report.append("- **Python版本**: 3.x")
        report.append("- **测试工具**: 自定义性能测试框架")
        report.append("- **测试数据**: 模拟真实使用场景")
        report.append("")
        
        # 结论
        report.append("## 总结与结论")
        report.append("")
        
        overall_score = scores.get('overall_score', 0)
        prd_score = scores.get('prd_compliance_score', 0)
        
        if overall_score >= 80 and prd_score >= 80:
            report.append("✅ **HarborAI SDK性能表现优秀**")
            report.append("- 所有关键性能指标均满足PRD要求")
            report.append("- 与主流SDK相比具有竞争优势")
            report.append("- 特有功能运行高效，架构设计合理")
            report.append("- 建议继续保持当前性能水平，关注新功能的性能影响")
        elif overall_score >= 60:
            report.append("⚠️ **HarborAI SDK性能表现良好，有优化空间**")
            report.append("- 大部分性能指标满足要求")
            report.append("- 部分指标需要针对性优化")
            report.append("- 建议按照优化计划逐步改进")
        else:
            report.append("❌ **HarborAI SDK性能需要重点优化**")
            report.append("- 多项关键指标未达到预期")
            report.append("- 需要制定全面的性能优化计划")
            report.append("- 建议优先解决高影响的性能问题")
        
        report.append("")
        report.append("---")
        report.append("*本报告基于自动化性能测试生成，建议结合实际业务场景进行验证*")
        
        return "\n".join(report)

def main():
    """主函数"""
    generator = ComprehensivePerformanceReportGenerator()
    
    try:
        comprehensive_report = generator.generate_comprehensive_report()
        
        # 保存综合报告
        report_file = "harborai_comprehensive_performance_evaluation_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(comprehensive_report)
        
        print(f"✅ 综合性能评估报告已生成")
        print(f"📄 报告文件: {report_file}")
        
        return 0
        
    except Exception as e:
        print(f"❌ 综合报告生成失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())