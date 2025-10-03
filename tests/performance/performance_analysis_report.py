#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI SDK 性能分析报告生成器

基于性能测试结果，生成详细的分析报告和优化建议
"""

import json
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

@dataclass
class PerformanceThreshold:
    """性能阈值定义"""
    name: str
    value: float
    unit: str
    comparison: str  # 'less_than', 'greater_than', 'equal_to'
    description: str

class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self, results_file: str):
        """初始化分析器"""
        self.results_file = results_file
        self.results = self._load_results()
        
        # PRD/TD 性能要求阈值
        self.prd_thresholds = [
            PerformanceThreshold("调用封装开销", 1.0, "ms", "less_than", "每次API调用的封装开销应小于1毫秒"),
            PerformanceThreshold("高并发成功率", 99.9, "%", "greater_than", "高并发场景下的成功率应大于99.9%"),
            PerformanceThreshold("内存泄漏", 10.0, "MB", "less_than", "长期运行时的内存泄漏应小于10MB"),
            PerformanceThreshold("初始化时间", 500.0, "ms", "less_than", "SDK初始化时间应小于500毫秒"),
            PerformanceThreshold("并发吞吐量", 100.0, "ops/s", "greater_than", "并发处理能力应大于100操作/秒")
        ]
    
    def _load_results(self) -> Dict[str, Any]:
        """加载测试结果"""
        try:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 加载结果文件失败: {e}")
            return {}
    
    def analyze_initialization_performance(self) -> Dict[str, Any]:
        """分析初始化性能"""
        init_data = self.results.get('initialization_overhead', {})
        
        analysis = {
            'summary': {},
            'bottlenecks': [],
            'recommendations': []
        }
        
        # 分析各模式的初始化时间
        for mode, metrics in init_data.items():
            avg_time = metrics.get('avg_ms', 0)
            max_time = metrics.get('max_ms', 0)
            
            analysis['summary'][mode] = {
                'avg_time_ms': avg_time,
                'max_time_ms': max_time,
                'meets_threshold': avg_time < 500.0,
                'performance_grade': self._grade_performance(avg_time, 200, 350, 500)
            }
            
            # 识别瓶颈
            if avg_time > 400:
                analysis['bottlenecks'].append(f"{mode}模式初始化时间过长: {avg_time:.1f}ms")
            
            if max_time > avg_time * 1.5:
                analysis['bottlenecks'].append(f"{mode}模式初始化时间不稳定，最大值是平均值的{max_time/avg_time:.1f}倍")
        
        # 生成优化建议
        if analysis['bottlenecks']:
            analysis['recommendations'].extend([
                "考虑延迟初始化非关键组件",
                "优化插件管理器的初始化流程",
                "减少启动时的配置验证开销",
                "使用连接池预热策略"
            ])
        
        return analysis
    
    def analyze_method_call_performance(self) -> Dict[str, Any]:
        """分析方法调用性能"""
        method_data = self.results.get('method_call_overhead', {})
        
        analysis = {
            'summary': {},
            'bottlenecks': [],
            'recommendations': []
        }
        
        # 分析各方法的调用开销
        for method, metrics in method_data.items():
            avg_time_us = metrics.get('avg_us', 0)
            avg_time_ms = avg_time_us / 1000
            p95_time_us = metrics.get('p95_us', 0)
            
            analysis['summary'][method] = {
                'avg_time_us': avg_time_us,
                'avg_time_ms': avg_time_ms,
                'p95_time_us': p95_time_us,
                'meets_threshold': avg_time_ms < 1.0,
                'performance_grade': self._grade_performance(avg_time_ms, 0.1, 0.5, 1.0)
            }
            
            # 识别瓶颈
            if avg_time_ms > 0.5:
                analysis['bottlenecks'].append(f"{method}调用开销较高: {avg_time_ms:.3f}ms")
            
            if p95_time_us > avg_time_us * 2:
                analysis['bottlenecks'].append(f"{method}调用时间不稳定，P95是平均值的{p95_time_us/avg_time_us:.1f}倍")
        
        # 生成优化建议
        if any(m['avg_time_ms'] > 0.3 for m in analysis['summary'].values()):
            analysis['recommendations'].extend([
                "优化参数验证逻辑，减少重复检查",
                "使用缓存机制存储常用配置",
                "减少方法调用链的深度",
                "考虑使用更高效的数据结构"
            ])
        
        return analysis
    
    def analyze_memory_performance(self) -> Dict[str, Any]:
        """分析内存性能"""
        memory_data = self.results.get('memory_usage', {})
        
        analysis = {
            'summary': {},
            'bottlenecks': [],
            'recommendations': []
        }
        
        # 基线内存
        baseline_mb = memory_data.get('baseline_mb', 0)
        
        # 客户端创建内存开销
        client_creation = memory_data.get('client_creation', {})
        for mode, metrics in client_creation.items():
            overhead = metrics.get('overhead_per_client_mb', 0)
            cleanup_efficiency = (metrics.get('after_mb', 0) - metrics.get('cleanup_mb', 0))
            
            analysis['summary'][f'{mode}_client'] = {
                'overhead_per_client_mb': overhead,
                'cleanup_efficiency_mb': cleanup_efficiency,
                'performance_grade': self._grade_performance(overhead, 1, 5, 10)
            }
        
        # 内存泄漏分析
        leak_test = memory_data.get('memory_leak_test', {})
        potential_leak = leak_test.get('potential_leak_mb', 0)
        
        analysis['summary']['memory_leak'] = {
            'potential_leak_mb': potential_leak,
            'meets_threshold': potential_leak < 10.0,
            'performance_grade': self._grade_performance(potential_leak, 2, 5, 10, reverse=True)
        }
        
        # 识别瓶颈
        if potential_leak > 5:
            analysis['bottlenecks'].append(f"存在潜在内存泄漏: {potential_leak:.2f}MB")
        
        for mode, metrics in client_creation.items():
            cleanup_efficiency = metrics.get('after_mb', 0) - metrics.get('cleanup_mb', 0)
            if cleanup_efficiency > 2:
                analysis['bottlenecks'].append(f"{mode}模式客户端清理不彻底: {cleanup_efficiency:.2f}MB")
        
        # 生成优化建议
        if analysis['bottlenecks']:
            analysis['recommendations'].extend([
                "检查对象引用，确保及时释放",
                "优化缓存策略，避免无限增长",
                "使用弱引用减少循环引用",
                "定期执行垃圾回收"
            ])
        
        return analysis
    
    def analyze_concurrent_performance(self) -> Dict[str, Any]:
        """分析并发性能"""
        concurrent_data = self.results.get('concurrent_performance', {})
        
        analysis = {
            'summary': {},
            'bottlenecks': [],
            'recommendations': [],
            'scalability': {}
        }
        
        # 分析各并发级别的性能
        concurrency_levels = []
        throughputs = []
        
        for level_key, metrics in concurrent_data.items():
            if level_key.startswith('concurrency_'):
                concurrency = int(level_key.split('_')[1])
                success_rate = metrics.get('success_rate', 0)
                ops_per_sec = metrics.get('operations_per_second', 0)
                avg_response_time = metrics.get('avg_response_time_ms', 0)
                
                concurrency_levels.append(concurrency)
                throughputs.append(ops_per_sec)
                
                analysis['summary'][f'concurrency_{concurrency}'] = {
                    'success_rate': success_rate,
                    'throughput_ops_per_sec': ops_per_sec,
                    'avg_response_time_ms': avg_response_time,
                    'meets_success_threshold': success_rate > 99.9,
                    'meets_throughput_threshold': ops_per_sec > 100,
                    'performance_grade': self._grade_concurrent_performance(success_rate, ops_per_sec)
                }
                
                # 识别瓶颈
                if success_rate < 99.9:
                    analysis['bottlenecks'].append(f"{concurrency}并发成功率不达标: {success_rate:.1f}%")
                
                if ops_per_sec < 100:
                    analysis['bottlenecks'].append(f"{concurrency}并发吞吐量不达标: {ops_per_sec:.1f}ops/s")
        
        # 可扩展性分析
        if len(concurrency_levels) >= 2:
            # 计算吞吐量增长率
            throughput_growth = []
            for i in range(1, len(throughputs)):
                growth = (throughputs[i] - throughputs[i-1]) / throughputs[i-1] * 100
                throughput_growth.append(growth)
            
            analysis['scalability'] = {
                'linear_scaling': all(growth > 0 for growth in throughput_growth),
                'avg_growth_rate': sum(throughput_growth) / len(throughput_growth) if throughput_growth else 0,
                'peak_throughput': max(throughputs),
                'optimal_concurrency': concurrency_levels[throughputs.index(max(throughputs))]
            }
        
        # 生成优化建议
        if analysis['bottlenecks']:
            analysis['recommendations'].extend([
                "优化线程池配置",
                "减少锁竞争和同步开销",
                "使用异步I/O提高并发能力",
                "优化资源池管理"
            ])
        
        return analysis
    
    def _grade_performance(self, value: float, excellent: float, good: float, acceptable: float, reverse: bool = False) -> str:
        """性能评级"""
        if not reverse:
            if value <= excellent:
                return "优秀"
            elif value <= good:
                return "良好"
            elif value <= acceptable:
                return "可接受"
            else:
                return "需要优化"
        else:
            if value >= excellent:
                return "优秀"
            elif value >= good:
                return "良好"
            elif value >= acceptable:
                return "可接受"
            else:
                return "需要优化"
    
    def _grade_concurrent_performance(self, success_rate: float, throughput: float) -> str:
        """并发性能评级"""
        if success_rate > 99.9 and throughput > 300:
            return "优秀"
        elif success_rate > 99.5 and throughput > 200:
            return "良好"
        elif success_rate > 99.0 and throughput > 100:
            return "可接受"
        else:
            return "需要优化"
    
    def check_prd_compliance(self) -> Dict[str, Any]:
        """检查PRD合规性"""
        compliance = {
            'overall_score': 0,
            'passed_checks': 0,
            'total_checks': len(self.prd_thresholds),
            'details': []
        }
        
        for threshold in self.prd_thresholds:
            check_result = self._check_single_threshold(threshold)
            compliance['details'].append(check_result)
            if check_result['passed']:
                compliance['passed_checks'] += 1
        
        compliance['overall_score'] = (compliance['passed_checks'] / compliance['total_checks']) * 100
        compliance['compliance_level'] = self._get_compliance_level(compliance['overall_score'])
        
        return compliance
    
    def _check_single_threshold(self, threshold: PerformanceThreshold) -> Dict[str, Any]:
        """检查单个阈值"""
        result = {
            'name': threshold.name,
            'threshold_value': threshold.value,
            'unit': threshold.unit,
            'description': threshold.description,
            'passed': False,
            'actual_value': None,
            'deviation': None
        }
        
        # 根据阈值类型获取实际值
        if threshold.name == "调用封装开销":
            # 取所有方法调用的平均开销
            method_data = self.results.get('method_call_overhead', {})
            if method_data:
                avg_times = [metrics.get('avg_us', 0) / 1000 for metrics in method_data.values()]
                result['actual_value'] = sum(avg_times) / len(avg_times) if avg_times else 0
        
        elif threshold.name == "高并发成功率":
            # 取最高并发级别的成功率
            concurrent_data = self.results.get('concurrent_performance', {})
            max_concurrency_key = max([k for k in concurrent_data.keys() if k.startswith('concurrency_')], 
                                    key=lambda x: int(x.split('_')[1]), default=None)
            if max_concurrency_key:
                result['actual_value'] = concurrent_data[max_concurrency_key].get('success_rate', 0)
        
        elif threshold.name == "内存泄漏":
            memory_data = self.results.get('memory_usage', {})
            leak_test = memory_data.get('memory_leak_test', {})
            result['actual_value'] = leak_test.get('potential_leak_mb', 0)
        
        elif threshold.name == "初始化时间":
            # 取所有模式的平均初始化时间
            init_data = self.results.get('initialization_overhead', {})
            if init_data:
                avg_times = [metrics.get('avg_ms', 0) for metrics in init_data.values()]
                result['actual_value'] = sum(avg_times) / len(avg_times) if avg_times else 0
        
        elif threshold.name == "并发吞吐量":
            # 取最高吞吐量
            concurrent_data = self.results.get('concurrent_performance', {})
            throughputs = [metrics.get('operations_per_second', 0) 
                          for metrics in concurrent_data.values()]
            result['actual_value'] = max(throughputs) if throughputs else 0
        
        # 检查是否通过阈值
        if result['actual_value'] is not None:
            if threshold.comparison == "less_than":
                result['passed'] = result['actual_value'] < threshold.value
                result['deviation'] = result['actual_value'] - threshold.value
            elif threshold.comparison == "greater_than":
                result['passed'] = result['actual_value'] > threshold.value
                result['deviation'] = threshold.value - result['actual_value']
        
        return result
    
    def _get_compliance_level(self, score: float) -> str:
        """获取合规等级"""
        if score >= 90:
            return "优秀"
        elif score >= 80:
            return "良好"
        elif score >= 70:
            return "可接受"
        else:
            return "需要改进"
    
    def generate_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """生成优化建议"""
        recommendations = []
        
        # 基于各项分析结果生成建议
        init_analysis = self.analyze_initialization_performance()
        method_analysis = self.analyze_method_call_performance()
        memory_analysis = self.analyze_memory_performance()
        concurrent_analysis = self.analyze_concurrent_performance()
        
        # 初始化优化建议
        if init_analysis['bottlenecks']:
            recommendations.append({
                'category': '初始化优化',
                'priority': 'high',
                'impact': 'medium',
                'effort': 'medium',
                'recommendations': init_analysis['recommendations'],
                'expected_improvement': '减少20-30%的启动时间'
            })
        
        # 方法调用优化建议
        if method_analysis['bottlenecks']:
            recommendations.append({
                'category': '方法调用优化',
                'priority': 'high',
                'impact': 'high',
                'effort': 'low',
                'recommendations': method_analysis['recommendations'],
                'expected_improvement': '减少50-70%的调用开销'
            })
        
        # 内存优化建议
        if memory_analysis['bottlenecks']:
            recommendations.append({
                'category': '内存管理优化',
                'priority': 'medium',
                'impact': 'medium',
                'effort': 'medium',
                'recommendations': memory_analysis['recommendations'],
                'expected_improvement': '减少内存泄漏，提高长期稳定性'
            })
        
        # 并发优化建议
        if concurrent_analysis['bottlenecks']:
            recommendations.append({
                'category': '并发性能优化',
                'priority': 'high',
                'impact': 'high',
                'effort': 'high',
                'recommendations': concurrent_analysis['recommendations'],
                'expected_improvement': '提高30-50%的并发处理能力'
            })
        
        return recommendations
    
    def generate_comprehensive_report(self) -> str:
        """生成综合分析报告"""
        report = []
        
        # 报告头部
        report.append("# HarborAI SDK 性能分析报告")
        report.append("=" * 60)
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 测试概览
        test_info = self.results.get('test_info', {})
        report.append("## 测试概览")
        report.append(f"- 测试开始时间: {test_info.get('start_time', 'N/A')}")
        report.append(f"- 测试结束时间: {test_info.get('end_time', 'N/A')}")
        report.append(f"- 总测试时长: {test_info.get('total_duration_seconds', 0):.1f}秒")
        
        system_info = test_info.get('system_info', {})
        report.append(f"- CPU核心数: {system_info.get('cpu_count', 'N/A')}")
        report.append(f"- 总内存: {system_info.get('memory_total_gb', 0):.1f}GB")
        report.append(f"- 平台: {system_info.get('platform', 'N/A')}")
        report.append("")
        
        # PRD合规性检查
        compliance = self.check_prd_compliance()
        report.append("## PRD合规性检查")
        report.append(f"- 总体得分: {compliance['overall_score']:.1f}%")
        report.append(f"- 合规等级: {compliance['compliance_level']}")
        report.append(f"- 通过检查: {compliance['passed_checks']}/{compliance['total_checks']}")
        report.append("")
        
        for detail in compliance['details']:
            status = "✅" if detail['passed'] else "❌"
            actual_val = detail['actual_value'] if detail['actual_value'] is not None else 0
            report.append(f"{status} {detail['name']}: {actual_val:.3f}{detail['unit']} "
                         f"(阈值: {detail['threshold_value']}{detail['unit']})")
        report.append("")
        
        # 各项性能分析
        analyses = [
            ("初始化性能分析", self.analyze_initialization_performance()),
            ("方法调用性能分析", self.analyze_method_call_performance()),
            ("内存性能分析", self.analyze_memory_performance()),
            ("并发性能分析", self.analyze_concurrent_performance())
        ]
        
        for title, analysis in analyses:
            report.append(f"## {title}")
            
            # 性能摘要
            if 'summary' in analysis:
                report.append("### 性能摘要")
                for item, metrics in analysis['summary'].items():
                    grade = metrics.get('performance_grade', 'N/A')
                    report.append(f"- {item}: {grade}")
                report.append("")
            
            # 瓶颈识别
            if analysis.get('bottlenecks'):
                report.append("### 识别的瓶颈")
                for bottleneck in analysis['bottlenecks']:
                    report.append(f"- ⚠️ {bottleneck}")
                report.append("")
            
            # 优化建议
            if analysis.get('recommendations'):
                report.append("### 优化建议")
                for rec in analysis['recommendations']:
                    report.append(f"- 💡 {rec}")
                report.append("")
        
        # 综合优化建议
        recommendations = self.generate_optimization_recommendations()
        if recommendations:
            report.append("## 综合优化建议")
            for i, rec in enumerate(recommendations, 1):
                report.append(f"### {i}. {rec['category']}")
                report.append(f"- 优先级: {rec['priority']}")
                report.append(f"- 影响程度: {rec['impact']}")
                report.append(f"- 实施难度: {rec['effort']}")
                report.append(f"- 预期改进: {rec['expected_improvement']}")
                report.append("- 具体建议:")
                for suggestion in rec['recommendations']:
                    report.append(f"  - {suggestion}")
                report.append("")
        
        # 结论和下一步行动
        report.append("## 结论和下一步行动")
        report.append("基于性能测试结果，HarborAI SDK在以下方面表现良好：")
        
        # 根据合规性得分给出结论
        if compliance['overall_score'] >= 80:
            report.append("- ✅ 整体性能表现优秀，满足大部分PRD要求")
        else:
            report.append("- ⚠️ 存在一些性能问题需要优化")
        
        report.append("")
        report.append("建议的下一步行动：")
        report.append("1. 优先解决高优先级的性能瓶颈")
        report.append("2. 实施具体的优化建议")
        report.append("3. 建立持续的性能监控机制")
        report.append("4. 定期进行性能回归测试")
        
        return "\n".join(report)

def main():
    """主函数"""
    results_file = "sdk_performance_results.json"
    
    if not os.path.exists(results_file):
        print(f"❌ 结果文件不存在: {results_file}")
        return 1
    
    analyzer = PerformanceAnalyzer(results_file)
    
    # 生成综合报告
    report = analyzer.generate_comprehensive_report()
    
    # 保存报告
    report_file = "harborai_performance_analysis_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("📊 HarborAI SDK 性能分析完成")
    print(f"📄 详细报告已保存到: {report_file}")
    
    # 打印关键结果
    compliance = analyzer.check_prd_compliance()
    print(f"\n🎯 PRD合规性得分: {compliance['overall_score']:.1f}% ({compliance['compliance_level']})")
    print(f"✅ 通过检查: {compliance['passed_checks']}/{compliance['total_checks']}")
    
    return 0

if __name__ == "__main__":
    exit(main())