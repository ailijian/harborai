#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI vs 原生 OpenAI 综合性能对比测试

此脚本对比原生 OpenAI 与 HarborAI 各模式的性能差异，
使用多个测试样本和多次运行来获得准确的统计数据。
"""

import os
import time
import statistics
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from harborai import HarborAI
from harborai.config.settings import get_settings

# 加载 .env 文件
load_dotenv()

# 测试配置
TEST_ROUNDS = 10  # 每个模式运行次数
TEST_SAMPLES = [
    {
        "name": "简单问答",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "请简单介绍一下你自己，不超过50字。"}
        ]
    },
    {
        "name": "代码生成",
        "messages": [
            {"role": "system", "content": "You are a programming assistant"},
            {"role": "user", "content": "请写一个Python函数来计算斐波那契数列的第n项。"}
        ]
    }
]


def print_separator(title: str):
    """打印分隔符"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_native_openai(sample: Dict[str, Any], round_num: int) -> Dict[str, Any]:
    """测试原生 OpenAI 调用"""
    print(f"  第 {round_num}/{TEST_ROUNDS} 轮测试...")
    
    # 从环境变量获取配置
    api_key = os.environ.get('DEEPSEEK_API_KEY')
    base_url = os.environ.get('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
    
    if not api_key:
        return {
            'success': False,
            'duration': 0,
            'error': '未找到 DEEPSEEK_API_KEY 环境变量',
            'content': '',
            'tokens': 0
        }
    
    start_time = time.time()
    
    try:
        # 创建原生 OpenAI 客户端
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        # 进行 API 调用
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=sample["messages"],
            max_tokens=200,
            temperature=0.7,
            stream=False
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 提取响应内容
        content = response.choices[0].message.content
        tokens = response.usage.total_tokens if response.usage else 0
        
        result = {
            'success': True,
            'duration': duration,
            'content': content,
            'tokens': tokens
        }
        
        print(f"    [成功] - 耗时: {duration:.3f}s, Tokens: {tokens}")
        return result
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        result = {
            'success': False,
            'duration': duration,
            'error': str(e),
            'content': '',
            'tokens': 0
        }
        
        print(f"    [失败] - 耗时: {duration:.3f}s, 错误: {e}")
        return result


def test_harborai_mode(mode: str, sample: Dict[str, Any], round_num: int) -> Dict[str, Any]:
    """测试 HarborAI 指定模式"""
    print(f"  第 {round_num}/{TEST_ROUNDS} 轮测试...")
    
    # 设置环境变量
    os.environ['HARBORAI_PERFORMANCE_MODE'] = mode
    if mode == 'fast':
        os.environ['HARBORAI_ENABLE_FAST_PATH'] = 'true'
    elif mode == 'full':
        os.environ['HARBORAI_ENABLE_FAST_PATH'] = 'false'
    else:  # balanced
        os.environ['HARBORAI_ENABLE_FAST_PATH'] = 'true'
    
    # 重新加载配置
    get_settings.cache_clear()
    
    # 从环境变量获取配置
    api_key = os.environ.get('DEEPSEEK_API_KEY')
    base_url = os.environ.get('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
    
    if not api_key:
        return {
            'success': False,
            'duration': 0,
            'error': '未找到 DEEPSEEK_API_KEY 环境变量',
            'content': '',
            'tokens': 0
        }
    
    start_time = time.time()
    
    try:
        # 创建 HarborAI 客户端
        client = HarborAI(
            api_key=api_key,
            base_url=base_url
        )
        
        # 进行 API 调用（同步方式）
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=sample["messages"],
            max_tokens=200,
            temperature=0.7
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 提取响应内容
        content = response.choices[0].message.content
        
        # 安全地获取 tokens 数量，避免 model_dump 错误
        tokens = 0
        if hasattr(response, 'usage') and response.usage:
            if hasattr(response.usage, 'total_tokens'):
                tokens = response.usage.total_tokens
            elif hasattr(response.usage, '__dict__'):
                tokens = getattr(response.usage, 'total_tokens', 0)
        
        result = {
            'success': True,
            'duration': duration,
            'content': content,
            'tokens': tokens
        }
        
        print(f"    [成功] - 耗时: {duration:.3f}s, Tokens: {tokens}")
        return result
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        result = {
            'success': False,
            'duration': duration,
            'error': str(e),
            'content': '',
            'tokens': 0
        }
        
        print(f"    [失败] - 耗时: {duration:.3f}s, 错误: {e}")
        return result


def run_performance_test(test_func, test_name: str, sample: Dict[str, Any]) -> Dict[str, Any]:
    """运行性能测试并计算统计数据"""
    print(f"\n--- {test_name} - {sample['name']} ---")
    
    results = []
    successful_results = []
    
    for round_num in range(1, TEST_ROUNDS + 1):
        if test_name == "原生 OpenAI":
            result = test_func(sample, round_num)
        else:
            # HarborAI 模式测试
            mode = test_name.split()[1].lower()  # 从 "HarborAI FULL" 提取 "full"
            result = test_func(mode, sample, round_num)
        
        results.append(result)
        if result['success']:
            successful_results.append(result)
        
        # 轮次间稍作等待
        if round_num < TEST_ROUNDS:
            time.sleep(0.5)
    
    # 计算统计数据
    if successful_results:
        durations = [r['duration'] for r in successful_results]
        tokens = [r['tokens'] for r in successful_results]
        
        stats = {
            'success_rate': len(successful_results) / len(results) * 100,
            'avg_duration': statistics.mean(durations),
            'std_duration': statistics.stdev(durations) if len(durations) > 1 else 0,
            'min_duration': min(durations),
            'max_duration': max(durations),
            'avg_tokens': statistics.mean(tokens) if tokens else 0,
            'total_tests': len(results),
            'successful_tests': len(successful_results)
        }
        
        print(f"  [结果] 统计结果:")
        print(f"    成功率: {stats['success_rate']:.1f}% ({stats['successful_tests']}/{stats['total_tests']})")
        print(f"    平均耗时: {stats['avg_duration']:.3f}s ± {stats['std_duration']:.3f}s")
        print(f"    耗时范围: {stats['min_duration']:.3f}s - {stats['max_duration']:.3f}s")
        if stats['avg_tokens'] > 0:
            print(f"    平均Tokens: {stats['avg_tokens']:.0f}")
    else:
        stats = {
            'success_rate': 0,
            'avg_duration': 0,
            'std_duration': 0,
            'min_duration': 0,
            'max_duration': 0,
            'avg_tokens': 0,
            'total_tests': len(results),
            'successful_tests': 0
        }
        print(f"  [错误] 所有测试均失败")
        if results:
            print(f"    最后错误: {results[-1]['error']}")
    
    return stats


def test_comprehensive_performance():
    """综合性能测试"""
    print_separator("HarborAI vs 原生 OpenAI 综合性能对比测试")
    
    # 检查环境变量
    if not os.environ.get('DEEPSEEK_API_KEY'):
        print("[错误] 未找到 DEEPSEEK_API_KEY 环境变量")
        print("请检查 .env 文件中的配置")
        return
    
    print(f"\n[配置] 测试配置:")
    print(f"  测试轮次: {TEST_ROUNDS}")
    print(f"  测试样本: {len(TEST_SAMPLES)}个")
    print(f"  样本类型: {', '.join([s['name'] for s in TEST_SAMPLES])}")
    
    # 存储所有测试结果
    all_results = {}
    
    # 测试每个样本
    for sample in TEST_SAMPLES:
        print_separator(f"测试样本: {sample['name']}")
        
        sample_results = {}
        
        # 1. 测试原生 OpenAI
        sample_results['原生 OpenAI'] = run_performance_test(
            test_native_openai, "原生 OpenAI", sample
        )
        
        # 2. 测试 HarborAI FULL 模式
        sample_results['HarborAI FULL'] = run_performance_test(
            test_harborai_mode, "HarborAI FULL", sample
        )
        
        # 3. 测试 HarborAI FAST 模式
        sample_results['HarborAI FAST'] = run_performance_test(
            test_harborai_mode, "HarborAI FAST", sample
        )
        
        # 4. 测试 HarborAI BALANCED 模式
        sample_results['HarborAI BALANCED'] = run_performance_test(
            test_harborai_mode, "HarborAI BALANCED", sample
        )
        
        all_results[sample['name']] = sample_results
    
    # 生成综合报告
    generate_performance_report(all_results)


def generate_performance_report(all_results: Dict[str, Dict[str, Dict[str, Any]]]):
    """生成性能报告"""
    print_separator("综合性能报告")
    
    # 按样本显示详细结果
    for sample_name, sample_results in all_results.items():
        print(f"\n[对比] {sample_name} - 详细对比:")
        print(f"{'模式':<20} {'成功率':<10} {'平均耗时':<12} {'标准差':<10} {'相对性能':<12}")
        print("-" * 70)
        
        # 以原生 OpenAI 为基准
        baseline_duration = sample_results.get('原生 OpenAI', {}).get('avg_duration', 0)
        
        for mode, stats in sample_results.items():
            success_rate = f"{stats['success_rate']:.1f}%"
            avg_duration = f"{stats['avg_duration']:.3f}s"
            std_duration = f"±{stats['std_duration']:.3f}s"
            
            if baseline_duration > 0 and stats['avg_duration'] > 0:
                relative_perf = f"{baseline_duration / stats['avg_duration']:.2f}x"
            else:
                relative_perf = "N/A"
            
            print(f"{mode:<20} {success_rate:<10} {avg_duration:<12} {std_duration:<10} {relative_perf:<12}")
    
    # 计算总体统计
    print(f"\n[统计] 总体性能统计:")
    
    modes = ['原生 OpenAI', 'HarborAI FULL', 'HarborAI FAST', 'HarborAI BALANCED']
    
    for mode in modes:
        all_durations = []
        all_success_rates = []
        
        for sample_results in all_results.values():
            if mode in sample_results:
                stats = sample_results[mode]
                if stats['success_rate'] > 0:
                    all_durations.append(stats['avg_duration'])
                all_success_rates.append(stats['success_rate'])
        
        if all_durations:
            avg_duration = statistics.mean(all_durations)
            avg_success_rate = statistics.mean(all_success_rates)
            
            print(f"  {mode}:")
            print(f"    平均成功率: {avg_success_rate:.1f}%")
            print(f"    平均响应时间: {avg_duration:.3f}s")
        else:
            print(f"  {mode}: 无有效数据")
    
    # 性能建议
    print(f"\n[建议] 配置建议:")
    print(f"""  
  [极速] 追求极致性能:
     HARBORAI_PERFORMANCE_MODE=fast
     HARBORAI_ENABLE_FAST_PATH=true
  
  [平衡] 平衡性能与功能:
     HARBORAI_PERFORMANCE_MODE=balanced
     HARBORAI_ENABLE_FAST_PATH=true
  
  [完整] 需要完整功能:
     HARBORAI_PERFORMANCE_MODE=full
     HARBORAI_ENABLE_FAST_PATH=false
  
  [监控] 监控与调试:
     HARBORAI_ENABLE_COST_TRACKING=true
     HARBORAI_LOG_LEVEL=INFO
  
  [基准] 性能基准:
     原生 OpenAI 作为性能基准进行对比
  """)


def main():
    """主函数"""
    # 检查 DeepSeek API 密钥
    if 'DEEPSEEK_API_KEY' not in os.environ:
        print("[警告] 请设置 DEEPSEEK_API_KEY 环境变量")
        print("   请检查 .env 文件中的配置")
        return
    
    # 运行综合性能测试
    test_comprehensive_performance()


if __name__ == "__main__":
    main()