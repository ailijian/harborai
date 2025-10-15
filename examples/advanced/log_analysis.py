#!/usr/bin/env python3
"""
HarborAI 高级日志分析工具

这个脚本提供了 HarborAI 日志的高级分析功能，包括：
1. 多维度日志过滤和查询
2. 日志类型统计和可视化
3. 性能分析和趋势监控
4. 错误模式识别和分析
5. 自定义报告生成
6. 交互式日志浏览
7. 日志数据导出和备份

作者: HarborAI Team
版本: 1.0.0
创建时间: 2024-01-01
"""

import subprocess
import time
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import re
from collections import defaultdict, Counter
import statistics


class LogAnalyzer:
    """高级日志分析器"""
    
    def __init__(self):
        """初始化分析器"""
        self.project_root = Path.cwd()
        self.view_logs_script = self.project_root / "view_logs.py"
        
        # 检查依赖
        if not self.view_logs_script.exists():
            raise FileNotFoundError(f"未找到 view_logs.py 脚本: {self.view_logs_script}")
    
    def _extract_json_content(self, output: str) -> Optional[str]:
        """从输出中提取JSON内容，忽略日志前缀
        
        Args:
            output: 原始输出内容
            
        Returns:
            Optional[str]: 提取的JSON内容，如果未找到返回None
        """
        if not output or not output.strip():
            return None
        
        # 查找JSON开始位置，优先查找数组，然后查找对象
        json_start = -1
        for char in ['[', '{']:
            pos = output.find(char)
            if pos != -1:
                if json_start == -1 or pos < json_start:
                    json_start = pos
        
        if json_start != -1:
            return output[json_start:].strip()
        
        return None
    
    def run_view_logs_command(self, args: List[str]) -> Tuple[bool, str, str]:
        """运行 view_logs.py 命令"""
        cmd = ["python", str(self.view_logs_script)] + args
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                encoding='utf-8', 
                errors='ignore',
                timeout=30
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "命令执行超时"
        except Exception as e:
            return False, "", f"执行命令时出错: {e}"
    
    def get_log_statistics(self, days: int = 7) -> Dict[str, Any]:
        """获取日志统计信息"""
        success, stdout, stderr = self.run_view_logs_command([
            "--stats", "--format", "json", "--days", str(days)
        ])
        
        if not success:
            print(f"❌ 获取统计信息失败: {stderr}")
            return {}
        
        try:
            # 提取JSON部分，忽略日志前缀
            json_content = self._extract_json_content(stdout)
            if json_content:
                return json.loads(json_content)
            else:
                print(f"❌ 未找到JSON内容")
                return {}
        except json.JSONDecodeError as e:
            print(f"❌ 解析统计信息失败: {e}")
            print(f"原始输出: {stdout[:200]}...")  # 显示前200字符用于调试
            return {}
    
    def get_logs_by_type(self, log_type: str, limit: int = 100, days: int = 7) -> List[Dict[str, Any]]:
        """按类型获取日志"""
        success, stdout, stderr = self.run_view_logs_command([
            "--type", log_type, 
            "--format", "json", 
            "--limit", str(limit),
            "--days", str(days)
        ])
        
        if not success:
            print(f"❌ 获取 {log_type} 日志失败: {stderr}")
            return []
        
        try:
            # 提取JSON部分，忽略日志前缀
            json_content = self._extract_json_content(stdout)
            if json_content:
                data = json.loads(json_content)
                # 处理不同的JSON结构
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    return data.get('data', data.get('logs', []))
                else:
                    return []
            else:
                print(f"❌ 未找到JSON内容")
                return []
        except json.JSONDecodeError as e:
            print(f"❌ 解析 {log_type} 日志失败: {e}")
            print(f"原始输出: {stdout[:200]}...")  # 显示前200字符用于调试
            return []
    
    def analyze_performance_trends(self, days: int = 7) -> Dict[str, Any]:
        """分析性能趋势"""
        print(f"🔍 分析最近 {days} 天的性能趋势...")
        
        # 获取响应日志
        response_logs = self.get_logs_by_type("response", limit=1000, days=days)
        
        if not response_logs:
            return {"error": "没有找到响应日志"}
        
        # 按模型分组分析
        model_performance = defaultdict(list)
        daily_performance = defaultdict(list)
        
        for log in response_logs:
            model = log.get('model', 'unknown')
            timestamp = log.get('timestamp', '')
            
            # 提取响应时间
            response_time = None
            if 'response_time' in log:
                response_time = log['response_time']
            elif 'metadata' in log and isinstance(log['metadata'], dict):
                response_time = log['metadata'].get('response_time')
            
            if response_time is not None:
                model_performance[model].append(response_time)
                
                # 按日期分组
                try:
                    date = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).date()
                    daily_performance[str(date)].append(response_time)
                except:
                    pass
        
        # 计算统计信息
        analysis = {
            "model_performance": {},
            "daily_trends": {},
            "overall_stats": {}
        }
        
        # 模型性能分析
        for model, times in model_performance.items():
            if times:
                analysis["model_performance"][model] = {
                    "count": len(times),
                    "avg_response_time": statistics.mean(times),
                    "median_response_time": statistics.median(times),
                    "min_response_time": min(times),
                    "max_response_time": max(times),
                    "std_dev": statistics.stdev(times) if len(times) > 1 else 0
                }
        
        # 每日趋势分析
        for date, times in daily_performance.items():
            if times:
                analysis["daily_trends"][date] = {
                    "count": len(times),
                    "avg_response_time": statistics.mean(times),
                    "median_response_time": statistics.median(times)
                }
        
        # 总体统计
        all_times = [t for times in model_performance.values() for t in times]
        if all_times:
            analysis["overall_stats"] = {
                "total_requests": len(all_times),
                "avg_response_time": statistics.mean(all_times),
                "median_response_time": statistics.median(all_times),
                "p95_response_time": sorted(all_times)[int(len(all_times) * 0.95)] if len(all_times) > 20 else max(all_times),
                "fastest_response": min(all_times),
                "slowest_response": max(all_times)
            }
        
        return analysis
    
    def analyze_error_patterns(self, days: int = 7) -> Dict[str, Any]:
        """分析错误模式"""
        print(f"🔍 分析最近 {days} 天的错误模式...")
        
        # 获取所有日志
        all_logs = self.get_logs_by_type("all", limit=1000, days=days)
        
        if not all_logs:
            return {"error": "没有找到日志"}
        
        # 错误分析
        error_patterns = {
            "error_by_model": defaultdict(int),
            "error_by_type": defaultdict(int),
            "error_by_hour": defaultdict(int),
            "common_errors": Counter(),
            "success_rate_by_model": {},
            "total_requests": 0,
            "total_errors": 0
        }
        
        model_stats = defaultdict(lambda: {"total": 0, "errors": 0})
        
        for log in all_logs:
            model = log.get('model', 'unknown')
            level = log.get('level', '').upper()
            message = log.get('message', '')
            timestamp = log.get('timestamp', '')
            
            model_stats[model]["total"] += 1
            error_patterns["total_requests"] += 1
            
            # 识别错误
            is_error = (
                level in ['ERROR', 'CRITICAL'] or
                'error' in message.lower() or
                'failed' in message.lower() or
                'exception' in message.lower()
            )
            
            if is_error:
                error_patterns["total_errors"] += 1
                model_stats[model]["errors"] += 1
                error_patterns["error_by_model"][model] += 1
                
                # 错误类型分类
                if 'timeout' in message.lower():
                    error_patterns["error_by_type"]["timeout"] += 1
                elif 'api' in message.lower():
                    error_patterns["error_by_type"]["api_error"] += 1
                elif 'auth' in message.lower() or 'key' in message.lower():
                    error_patterns["error_by_type"]["auth_error"] += 1
                elif 'rate' in message.lower() or 'limit' in message.lower():
                    error_patterns["error_by_type"]["rate_limit"] += 1
                else:
                    error_patterns["error_by_type"]["other"] += 1
                
                # 按小时统计
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    hour = dt.hour
                    error_patterns["error_by_hour"][hour] += 1
                except:
                    pass
                
                # 常见错误消息
                error_patterns["common_errors"][message[:100]] += 1
        
        # 计算成功率
        for model, stats in model_stats.items():
            if stats["total"] > 0:
                success_rate = (stats["total"] - stats["errors"]) / stats["total"] * 100
                error_patterns["success_rate_by_model"][model] = {
                    "total_requests": stats["total"],
                    "errors": stats["errors"],
                    "success_rate": success_rate
                }
        
        return dict(error_patterns)
    
    def generate_comprehensive_report(self, days: int = 7) -> Dict[str, Any]:
        """生成综合报告"""
        print(f"📊 生成最近 {days} 天的综合分析报告...")
        
        report = {
            "report_info": {
                "generated_at": datetime.now().isoformat(),
                "analysis_period_days": days,
                "report_version": "1.0.0"
            }
        }
        
        # 基础统计
        print("  📈 获取基础统计信息...")
        report["basic_stats"] = self.get_log_statistics(days)
        
        # 性能分析
        print("  ⚡ 分析性能趋势...")
        report["performance_analysis"] = self.analyze_performance_trends(days)
        
        # 错误分析
        print("  🚨 分析错误模式...")
        report["error_analysis"] = self.analyze_error_patterns(days)
        
        # 使用模式分析
        print("  📱 分析使用模式...")
        report["usage_patterns"] = self.analyze_usage_patterns(days)
        
        return report
    
    def analyze_usage_patterns(self, days: int = 7) -> Dict[str, Any]:
        """分析使用模式"""
        all_logs = self.get_logs_by_type("all", limit=1000, days=days)
        
        if not all_logs:
            return {"error": "没有找到日志"}
        
        patterns = {
            "hourly_distribution": defaultdict(int),
            "daily_distribution": defaultdict(int),
            "model_popularity": defaultdict(int),
            "request_size_distribution": defaultdict(int),
            "peak_hours": [],
            "busiest_days": []
        }
        
        for log in all_logs:
            timestamp = log.get('timestamp', '')
            model = log.get('model', 'unknown')
            
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                hour = dt.hour
                day = dt.strftime('%A')
                
                patterns["hourly_distribution"][hour] += 1
                patterns["daily_distribution"][day] += 1
                patterns["model_popularity"][model] += 1
                
                # 请求大小分析（基于消息长度）
                message = log.get('message', '')
                if len(message) < 100:
                    patterns["request_size_distribution"]["small"] += 1
                elif len(message) < 500:
                    patterns["request_size_distribution"]["medium"] += 1
                else:
                    patterns["request_size_distribution"]["large"] += 1
                    
            except:
                continue
        
        # 找出峰值时间
        if patterns["hourly_distribution"]:
            sorted_hours = sorted(patterns["hourly_distribution"].items(), key=lambda x: x[1], reverse=True)
            patterns["peak_hours"] = sorted_hours[:3]
        
        # 找出最忙的日子
        if patterns["daily_distribution"]:
            sorted_days = sorted(patterns["daily_distribution"].items(), key=lambda x: x[1], reverse=True)
            patterns["busiest_days"] = sorted_days[:3]
        
        return dict(patterns)
    
    def interactive_log_browser(self):
        """交互式日志浏览器"""
        print("🔍 进入交互式日志浏览模式")
        print("=" * 60)
        print("可用命令:")
        print("  1. stats [days] - 显示统计信息")
        print("  2. errors [days] - 显示错误分析")
        print("  3. performance [days] - 显示性能分析")
        print("  4. model <model_name> [days] - 显示特定模型的日志")
        print("  5. recent [limit] - 显示最近的日志")
        print("  6. search <keyword> - 搜索包含关键词的日志")
        print("  7. export <filename> - 导出分析报告")
        print("  8. help - 显示帮助")
        print("  9. quit - 退出")
        print("=" * 60)
        
        while True:
            try:
                command = input("\n🔍 请输入命令: ").strip().lower()
                
                if command == "quit" or command == "exit":
                    print("👋 退出交互式浏览器")
                    break
                elif command == "help":
                    self.show_interactive_help()
                elif command.startswith("stats"):
                    parts = command.split()
                    days = int(parts[1]) if len(parts) > 1 else 7
                    self.show_interactive_stats(days)
                elif command.startswith("errors"):
                    parts = command.split()
                    days = int(parts[1]) if len(parts) > 1 else 7
                    self.show_interactive_errors(days)
                elif command.startswith("performance"):
                    parts = command.split()
                    days = int(parts[1]) if len(parts) > 1 else 7
                    self.show_interactive_performance(days)
                elif command.startswith("model"):
                    parts = command.split()
                    if len(parts) < 2:
                        print("❌ 请指定模型名称")
                        continue
                    model_name = parts[1]
                    days = int(parts[2]) if len(parts) > 2 else 7
                    self.show_model_logs(model_name, days)
                elif command.startswith("recent"):
                    parts = command.split()
                    limit = int(parts[1]) if len(parts) > 1 else 10
                    self.show_recent_logs(limit)
                elif command.startswith("search"):
                    parts = command.split(maxsplit=1)
                    if len(parts) < 2:
                        print("❌ 请指定搜索关键词")
                        continue
                    keyword = parts[1]
                    self.search_logs(keyword)
                elif command.startswith("export"):
                    parts = command.split()
                    filename = parts[1] if len(parts) > 1 else f"log_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    self.export_report(filename)
                else:
                    print("❌ 未知命令，输入 'help' 查看帮助")
                    
            except KeyboardInterrupt:
                print("\n👋 退出交互式浏览器")
                break
            except Exception as e:
                print(f"❌ 执行命令时出错: {e}")
    
    def show_interactive_help(self):
        """显示交互式帮助"""
        print("\n📚 交互式日志浏览器帮助")
        print("=" * 50)
        print("命令格式:")
        print("  stats [days]           - 显示指定天数的统计信息 (默认7天)")
        print("  errors [days]          - 显示错误分析 (默认7天)")
        print("  performance [days]     - 显示性能分析 (默认7天)")
        print("  model <name> [days]    - 显示特定模型的日志")
        print("  recent [limit]         - 显示最近的日志 (默认10条)")
        print("  search <keyword>       - 搜索包含关键词的日志")
        print("  export [filename]      - 导出完整分析报告")
        print("  help                   - 显示此帮助信息")
        print("  quit/exit              - 退出浏览器")
        print("\n示例:")
        print("  stats 3                - 显示最近3天的统计")
        print("  model deepseek-chat 1  - 显示deepseek-chat模型最近1天的日志")
        print("  search error           - 搜索包含'error'的日志")
        print("  export my_report.json  - 导出报告到my_report.json")
    
    def show_interactive_stats(self, days: int):
        """显示交互式统计信息"""
        stats = self.get_log_statistics(days)
        if stats:
            print(f"\n📊 最近 {days} 天的统计信息:")
            print(json.dumps(stats, indent=2, ensure_ascii=False))
        else:
            print("❌ 无法获取统计信息")
    
    def show_interactive_errors(self, days: int):
        """显示交互式错误分析"""
        errors = self.analyze_error_patterns(days)
        if errors and "error" not in errors:
            print(f"\n🚨 最近 {days} 天的错误分析:")
            print(f"总请求数: {errors.get('total_requests', 0)}")
            print(f"总错误数: {errors.get('total_errors', 0)}")
            
            if errors.get('total_requests', 0) > 0:
                error_rate = errors.get('total_errors', 0) / errors.get('total_requests', 1) * 100
                print(f"错误率: {error_rate:.2f}%")
            
            print("\n按模型分组的错误:")
            for model, count in errors.get('error_by_model', {}).items():
                print(f"  {model}: {count}")
            
            print("\n错误类型分布:")
            for error_type, count in errors.get('error_by_type', {}).items():
                print(f"  {error_type}: {count}")
        else:
            print("❌ 无法获取错误分析或没有错误数据")
    
    def show_interactive_performance(self, days: int):
        """显示交互式性能分析"""
        perf = self.analyze_performance_trends(days)
        if perf and "error" not in perf:
            print(f"\n⚡ 最近 {days} 天的性能分析:")
            
            overall = perf.get('overall_stats', {})
            if overall:
                print(f"总请求数: {overall.get('total_requests', 0)}")
                print(f"平均响应时间: {overall.get('avg_response_time', 0):.3f}s")
                print(f"中位数响应时间: {overall.get('median_response_time', 0):.3f}s")
                print(f"95%分位数: {overall.get('p95_response_time', 0):.3f}s")
            
            print("\n按模型分组的性能:")
            for model, stats in perf.get('model_performance', {}).items():
                print(f"  {model}:")
                print(f"    请求数: {stats.get('count', 0)}")
                print(f"    平均响应时间: {stats.get('avg_response_time', 0):.3f}s")
                print(f"    中位数: {stats.get('median_response_time', 0):.3f}s")
        else:
            print("❌ 无法获取性能分析或没有性能数据")
    
    def show_model_logs(self, model_name: str, days: int):
        """显示特定模型的日志"""
        success, stdout, stderr = self.run_view_logs_command([
            "--model", model_name,
            "--days", str(days),
            "--limit", "10"
        ])
        
        if success:
            print(f"\n📱 模型 {model_name} 最近 {days} 天的日志:")
            print(stdout)
        else:
            print(f"❌ 获取模型 {model_name} 的日志失败: {stderr}")
    
    def show_recent_logs(self, limit: int):
        """显示最近的日志"""
        success, stdout, stderr = self.run_view_logs_command([
            "--limit", str(limit)
        ])
        
        if success:
            print(f"\n📋 最近 {limit} 条日志:")
            print(stdout)
        else:
            print(f"❌ 获取最近日志失败: {stderr}")
    
    def search_logs(self, keyword: str):
        """搜索日志"""
        # 获取所有日志并搜索
        all_logs = self.get_logs_by_type("all", limit=500, days=7)
        
        matching_logs = []
        for log in all_logs:
            message = log.get('message', '').lower()
            if keyword.lower() in message:
                matching_logs.append(log)
        
        if matching_logs:
            print(f"\n🔍 找到 {len(matching_logs)} 条包含 '{keyword}' 的日志:")
            for i, log in enumerate(matching_logs[:10], 1):  # 只显示前10条
                timestamp = log.get('timestamp', 'N/A')
                model = log.get('model', 'unknown')
                message = log.get('message', '')[:100]  # 截断长消息
                print(f"  [{i}] {timestamp} | {model} | {message}")
            
            if len(matching_logs) > 10:
                print(f"  ... 还有 {len(matching_logs) - 10} 条匹配的日志")
        else:
            print(f"❌ 没有找到包含 '{keyword}' 的日志")
    
    def export_report(self, filename: str):
        """导出分析报告"""
        try:
            print(f"📤 正在生成并导出报告到 {filename}...")
            report = self.generate_comprehensive_report(days=7)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 报告已成功导出到 {filename}")
            print(f"📊 报告包含:")
            print(f"  - 基础统计信息")
            print(f"  - 性能分析")
            print(f"  - 错误模式分析")
            print(f"  - 使用模式分析")
            
        except Exception as e:
            print(f"❌ 导出报告失败: {e}")


def run_demo_suite():
    """运行演示套件"""
    print("🚀 HarborAI 高级日志分析工具演示套件")
    print("=" * 60)
    
    analyzer = LogAnalyzer()
    
    demos = [
        {
            "name": "基础日志查看功能",
            "func": lambda: demo_basic_log_viewing(analyzer)
        },
        {
            "name": "性能趋势分析",
            "func": lambda: demo_performance_analysis(analyzer)
        },
        {
            "name": "错误模式识别",
            "func": lambda: demo_error_analysis(analyzer)
        },
        {
            "name": "使用模式分析",
            "func": lambda: demo_usage_patterns(analyzer)
        },
        {
            "name": "综合报告生成",
            "func": lambda: demo_comprehensive_report(analyzer)
        }
    ]
    
    for i, demo in enumerate(demos, 1):
        print(f"\n📋 演示 {i}/{len(demos)}: {demo['name']}")
        print("-" * 40)
        
        try:
            demo['func']()
        except Exception as e:
            print(f"❌ 演示失败: {e}")
        
        if i < len(demos):
            input("\n按 Enter 键继续下一个演示...")
    
    print(f"\n{'='*60}")
    print("🎉 演示套件完成！")
    print("💡 提示: 使用 --interactive 参数启动交互式浏览器")


def demo_log_types(analyzer: LogAnalyzer):
    """演示日志类型分析"""
    print("📊 日志类型分析演示")
    print("-" * 40)
    
    # 获取各种类型的日志
    log_types = ['request', 'response', 'paired']
    
    for log_type in log_types:
        print(f"\n🔍 分析 {log_type.upper()} 类型日志:")
        logs = analyzer.get_logs_by_type(log_type, limit=5)
        
        if logs:
            print(f"   ✅ 找到 {len(logs)} 条 {log_type} 日志")
            # 显示最新的一条日志摘要
            latest = logs[0] if logs else None
            if latest:
                print(f"   📝 最新记录: {latest.get('timestamp', 'N/A')}")
                print(f"   🤖 模型: {latest.get('model', 'unknown')}")
                print(f"   🏢 提供商: {latest.get('provider', 'unknown')}")
        else:
            print(f"   ❌ 未找到 {log_type} 类型的日志")
    
    print("\n✅ 日志类型分析完成")


def demo_basic_log_viewing(analyzer: LogAnalyzer):
    """演示基础日志查看功能"""
    print("🔍 演示基础日志查看功能...")
    
    # 显示帮助信息
    success, stdout, stderr = analyzer.run_view_logs_command(["--help"])
    if success:
        print("📚 view_logs.py 帮助信息:")
        print(stdout[:500] + "..." if len(stdout) > 500 else stdout)
    
    # 显示不同类型的日志
    log_types = ["all", "request", "response", "paired"]
    
    for log_type in log_types:
        print(f"\n📋 显示 {log_type} 类型的日志 (前3条):")
        success, stdout, stderr = analyzer.run_view_logs_command([
            "--type", log_type, "--limit", "3"
        ])
        if success and stdout.strip():
            print(stdout)
        else:
            print(f"  没有找到 {log_type} 类型的日志")


def demo_performance_analysis(analyzer: LogAnalyzer):
    """演示性能分析"""
    print("⚡ 演示性能趋势分析...")
    
    analysis = analyzer.analyze_performance_trends(days=7)
    
    if "error" in analysis:
        print(f"❌ {analysis['error']}")
        return
    
    overall = analysis.get('overall_stats', {})
    if overall:
        print("📊 总体性能统计:")
        print(f"  总请求数: {overall.get('total_requests', 0)}")
        print(f"  平均响应时间: {overall.get('avg_response_time', 0):.3f}s")
        print(f"  中位数响应时间: {overall.get('median_response_time', 0):.3f}s")
        print(f"  95%分位数: {overall.get('p95_response_time', 0):.3f}s")
        print(f"  最快响应: {overall.get('fastest_response', 0):.3f}s")
        print(f"  最慢响应: {overall.get('slowest_response', 0):.3f}s")
    
    model_perf = analysis.get('model_performance', {})
    if model_perf:
        print("\n📱 按模型分组的性能:")
        for model, stats in list(model_perf.items())[:5]:  # 只显示前5个
            print(f"  {model}:")
            print(f"    请求数: {stats.get('count', 0)}")
            print(f"    平均响应时间: {stats.get('avg_response_time', 0):.3f}s")
            print(f"    标准差: {stats.get('std_dev', 0):.3f}s")


def demo_error_analysis(analyzer: LogAnalyzer):
    """演示错误分析"""
    print("🚨 演示错误模式识别...")
    
    analysis = analyzer.analyze_error_patterns(days=7)
    
    if "error" in analysis:
        print(f"❌ {analysis['error']}")
        return
    
    print("📊 错误统计概览:")
    print(f"  总请求数: {analysis.get('total_requests', 0)}")
    print(f"  总错误数: {analysis.get('total_errors', 0)}")
    
    if analysis.get('total_requests', 0) > 0:
        error_rate = analysis.get('total_errors', 0) / analysis.get('total_requests', 1) * 100
        print(f"  错误率: {error_rate:.2f}%")
    
    # 按模型分组的成功率
    success_rates = analysis.get('success_rate_by_model', {})
    if success_rates:
        print("\n📱 按模型分组的成功率:")
        for model, stats in list(success_rates.items())[:5]:
            print(f"  {model}: {stats.get('success_rate', 0):.1f}% ({stats.get('errors', 0)}/{stats.get('total_requests', 0)})")
    
    # 错误类型分布
    error_types = analysis.get('error_by_type', {})
    if error_types:
        print("\n🏷️ 错误类型分布:")
        for error_type, count in error_types.items():
            print(f"  {error_type}: {count}")


def demo_usage_patterns(analyzer: LogAnalyzer):
    """演示使用模式分析"""
    print("📱 演示使用模式分析...")
    
    patterns = analyzer.analyze_usage_patterns(days=7)
    
    if "error" in patterns:
        print(f"❌ {patterns['error']}")
        return
    
    # 模型流行度
    model_popularity = patterns.get('model_popularity', {})
    if model_popularity:
        print("📊 模型使用频率:")
        sorted_models = sorted(model_popularity.items(), key=lambda x: x[1], reverse=True)
        for model, count in sorted_models[:5]:
            print(f"  {model}: {count} 次")
    
    # 峰值时间
    peak_hours = patterns.get('peak_hours', [])
    if peak_hours:
        print("\n⏰ 使用峰值时间:")
        for hour, count in peak_hours:
            print(f"  {hour:02d}:00 - {count} 次请求")
    
    # 最忙的日子
    busiest_days = patterns.get('busiest_days', [])
    if busiest_days:
        print("\n📅 最忙的日子:")
        for day, count in busiest_days:
            print(f"  {day}: {count} 次请求")


def demo_comprehensive_report(analyzer: LogAnalyzer):
    """演示综合报告生成"""
    print("📊 演示综合报告生成...")
    
    report = analyzer.generate_comprehensive_report(days=3)  # 使用较短时间以加快演示
    
    print("✅ 综合报告生成完成！")
    print("\n📋 报告包含以下部分:")
    
    for section, data in report.items():
        if section == "report_info":
            continue
        
        print(f"  📌 {section}")
        if isinstance(data, dict) and data:
            # 显示每个部分的简要信息
            if "total_requests" in data:
                print(f"     总请求数: {data['total_requests']}")
            if "total_errors" in data:
                print(f"     总错误数: {data['total_errors']}")
            if "overall_stats" in data and data["overall_stats"]:
                stats = data["overall_stats"]
                if "avg_response_time" in stats:
                    print(f"     平均响应时间: {stats['avg_response_time']:.3f}s")
    
    # 保存报告示例
    filename = f"demo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n💾 演示报告已保存到: {filename}")
    except Exception as e:
        print(f"❌ 保存报告失败: {e}")


def demo_new_logging_features(analyzer: LogAnalyzer):
    """演示新的日志功能特性"""
    print("🆕 演示新的日志功能特性...")
    
    # 演示布局模式
    print("\n📐 演示布局模式:")
    layouts = ["classic", "enhanced"]
    
    for layout in layouts:
        print(f"\n  🎨 {layout.upper()} 布局模式:")
        success, stdout, stderr = analyzer.run_view_logs_command([
            "--layout", layout, "--limit", "2"
        ])
        if success and stdout.strip():
            print(stdout)
        else:
            print(f"    没有找到日志数据")
    
    # 演示 trace_id 查询 (hb_ 前缀)
    print("\n🔍 演示 trace_id 查询功能:")
    
    # 首先获取一些 trace_id
    success, stdout, stderr = analyzer.run_view_logs_command([
        "--limit", "5", "--json"
    ])
    
    if success and stdout.strip():
        try:
            import json
            # 提取JSON部分，忽略日志前缀
            json_content = analyzer._extract_json_content(stdout)
            if json_content:
                logs = json.loads(json_content)
                # 确保logs是列表格式
                if isinstance(logs, dict):
                    logs = logs.get('data', logs.get('logs', []))
                elif not isinstance(logs, list):
                    logs = []
            else:
                logs = []
            trace_ids = []
            
            for log in logs:
                if isinstance(log, dict) and 'trace_id' in log:
                    trace_id = log['trace_id']
                    if trace_id and trace_id.startswith('hb_'):
                        trace_ids.append(trace_id)
            
            if trace_ids:
                # 使用第一个 trace_id 进行查询演示
                test_trace_id = trace_ids[0]
                print(f"  🎯 查询 trace_id: {test_trace_id}")
                
                success, stdout, stderr = analyzer.run_view_logs_command([
                    "--trace-id", test_trace_id
                ])
                
                if success and stdout.strip():
                    print(stdout)
                else:
                    print(f"    未找到 trace_id {test_trace_id} 的相关日志")
            else:
                print("    没有找到带有 hb_ 前缀的 trace_id")
                
        except json.JSONDecodeError:
            print("    JSON 解析失败")
    else:
        print("    无法获取日志数据进行 trace_id 演示")
    
    # 演示配对显示
    print("\n👥 演示配对显示功能:")
    success, stdout, stderr = analyzer.run_view_logs_command([
        "--type", "paired", "--limit", "3"
    ])
    if success and stdout.strip():
        print(stdout)
    else:
        print("    没有找到配对的日志数据")
    
    # 演示 JSON 输出
    print("\n📄 演示 JSON 格式输出:")
    success, stdout, stderr = analyzer.run_view_logs_command([
        "--json", "--limit", "2"
    ])
    if success and stdout.strip():
        try:
            import json
            # 提取JSON部分，忽略日志前缀
            json_content = analyzer._extract_json_content(stdout)
            if json_content:
                logs = json.loads(json_content)
                # 确保logs是列表格式
                if isinstance(logs, dict):
                    logs = logs.get('data', logs.get('logs', []))
                elif not isinstance(logs, list):
                    logs = []
            else:
                logs = []
            print(f"    获取到 {len(logs)} 条日志记录")
            if logs:
                print("    示例日志结构:")
                sample_log = logs[0]
                for key in ['timestamp', 'level', 'trace_id', 'type']:
                    if key in sample_log:
                        print(f"      {key}: {sample_log[key]}")
        except json.JSONDecodeError:
            print("    JSON 解析失败")
            print(stdout[:200] + "..." if len(stdout) > 200 else stdout)
    else:
        print("    无法获取 JSON 格式的日志数据")


def demo_trace_id_optimization():
    """演示 trace_id 优化功能"""
    print("🔧 演示 trace_id 优化功能...")
    
    # 检查最近的日志文件中是否有新格式的 trace_id
    import os
    import glob
    from datetime import datetime, timedelta
    
    log_dir = "logs"
    if not os.path.exists(log_dir):
        print("    日志目录不存在，无法演示 trace_id 优化")
        return
    
    # 查找最近几天的日志文件
    recent_files = []
    for i in range(3):  # 检查最近3天
        date_str = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
        pattern = f"{log_dir}/harborai_{date_str}.jsonl"
        files = glob.glob(pattern)
        recent_files.extend(files)
    
    if not recent_files:
        print("    没有找到最近的日志文件")
        return
    
    # 统计 trace_id 格式
    hb_prefix_count = 0
    old_format_count = 0
    total_count = 0
    
    for file_path in recent_files[:2]:  # 只检查前2个文件
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            import json
                            log_entry = json.loads(line)
                            if 'trace_id' in log_entry:
                                trace_id = log_entry['trace_id']
                                total_count += 1
                                if trace_id and trace_id.startswith('hb_'):
                                    hb_prefix_count += 1
                                else:
                                    old_format_count += 1
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"    读取文件 {file_path} 时出错: {e}")
    
    if total_count > 0:
        print(f"    📊 trace_id 格式统计:")
        print(f"      总数: {total_count}")
        print(f"      新格式 (hb_ 前缀): {hb_prefix_count} ({hb_prefix_count/total_count*100:.1f}%)")
        print(f"      旧格式: {old_format_count} ({old_format_count/total_count*100:.1f}%)")
        
        if hb_prefix_count > 0:
            print("    ✅ 检测到新的 hb_ 前缀 trace_id 格式")
        else:
            print("    ⚠️ 未检测到新的 hb_ 前缀 trace_id 格式")
    else:
        print("    没有找到有效的 trace_id 数据")


def run_demo_suite():
    """运行完整的演示套件"""
    print("🚀 启动 HarborAI 高级日志分析演示套件")
    print("=" * 60)
    
    analyzer = LogAnalyzer()
    
    # 基础功能演示
    demo_log_types(analyzer)
    print("\n" + "=" * 60)
    
    # 性能分析演示
    demo_performance_analysis(analyzer)
    print("\n" + "=" * 60)
    
    # 错误分析演示
    demo_error_analysis(analyzer)
    print("\n" + "=" * 60)
    
    # 使用模式分析演示
    demo_usage_patterns(analyzer)
    print("\n" + "=" * 60)
    
    # 新功能演示
    demo_new_logging_features(analyzer)
    print("\n" + "=" * 60)
    
    # trace_id 优化演示
    demo_trace_id_optimization()
    print("\n" + "=" * 60)
    
    # 综合报告演示
    demo_comprehensive_report(analyzer)
    
    print("\n🎉 演示套件运行完成！")
    print("💡 提示: 使用 --interactive 参数启动交互式日志浏览器")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="HarborAI 高级日志分析工具")
    parser.add_argument("--interactive", "-i", action="store_true", help="启动交互式日志浏览器")
    parser.add_argument("--demo", "-d", action="store_true", help="运行演示套件")
    parser.add_argument("--new-features", action="store_true", help="演示新的日志功能特性")
    parser.add_argument("--trace-id-demo", action="store_true", help="演示 trace_id 优化功能")
    parser.add_argument("--stats", "-s", type=int, default=7, help="显示指定天数的统计信息")
    parser.add_argument("--performance", "-p", type=int, help="分析指定天数的性能趋势")
    parser.add_argument("--errors", "-e", type=int, help="分析指定天数的错误模式")
    parser.add_argument("--report", "-r", help="生成综合报告并保存到指定文件")
    parser.add_argument("--days", type=int, default=7, help="分析天数 (默认7天)")
    
    args = parser.parse_args()
    
    try:
        analyzer = LogAnalyzer()
        
        if args.interactive:
            analyzer.interactive_log_browser()
        elif args.demo:
            run_demo_suite()
        elif args.new_features:
            demo_new_logging_features(analyzer)
        elif args.trace_id_demo:
            demo_trace_id_optimization()
        elif args.performance is not None:
            analysis = analyzer.analyze_performance_trends(args.performance)
            print(json.dumps(analysis, indent=2, ensure_ascii=False))
        elif args.errors is not None:
            analysis = analyzer.analyze_error_patterns(args.errors)
            print(json.dumps(analysis, indent=2, ensure_ascii=False))
        elif args.report:
            report = analyzer.generate_comprehensive_report(args.days)
            with open(args.report, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"✅ 报告已保存到: {args.report}")
        else:
            # 默认显示统计信息
            stats = analyzer.get_log_statistics(args.stats)
            if stats:
                print(json.dumps(stats, indent=2, ensure_ascii=False))
            else:
                print("❌ 无法获取统计信息")
                
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
        print("💡 请确保在 HarborAI 项目根目录下运行此脚本")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️ 程序被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()