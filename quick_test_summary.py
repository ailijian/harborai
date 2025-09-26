#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试摘要脚本
"""

import subprocess
import sys
from pathlib import Path

def quick_test(test_file):
    """快速测试单个文件"""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', str(test_file), '--tb=no', '-q'],
            capture_output=True,
            text=True,
            timeout=30,  # 30秒超时
            cwd=r'E:\project\harborai'
        )
        
        # 解析输出获取测试结果
        output = result.stdout + result.stderr
        if 'passed' in output and result.returncode == 0:
            return 'PASS'
        elif 'ERRORS' in output or 'ImportError' in output:
            return 'IMPORT_ERROR'
        elif 'FAILED' in output:
            return 'FAILED'
        else:
            return 'UNKNOWN'
    except subprocess.TimeoutExpired:
        return 'TIMEOUT'
    except Exception as e:
        return f'ERROR: {str(e)}'

def main():
    """主函数"""
    functional_dir = Path(r'E:\project\harborai\tests\functional')
    test_files = list(functional_dir.glob('test_*.py'))
    
    # 排除备份文件
    test_files = [f for f in test_files if 'backup' not in f.name]
    
    print(f"快速测试摘要 - 共 {len(test_files)} 个测试文件")
    print("=" * 60)
    
    results = {}
    for test_file in sorted(test_files):
        print(f"测试 {test_file.name}...", end=' ')
        status = quick_test(test_file)
        results[test_file.name] = status
        
        if status == 'PASS':
            print("✅ 通过")
        elif status == 'IMPORT_ERROR':
            print("❌ 导入错误")
        elif status == 'FAILED':
            print("❌ 测试失败")
        elif status == 'TIMEOUT':
            print("⏰ 超时")
        else:
            print(f"❓ {status}")
    
    # 统计结果
    print("\n" + "=" * 60)
    print("测试结果统计:")
    
    pass_count = sum(1 for status in results.values() if status == 'PASS')
    fail_count = sum(1 for status in results.values() if status == 'FAILED')
    import_error_count = sum(1 for status in results.values() if status == 'IMPORT_ERROR')
    timeout_count = sum(1 for status in results.values() if status == 'TIMEOUT')
    other_count = len(results) - pass_count - fail_count - import_error_count - timeout_count
    
    print(f"✅ 通过: {pass_count}")
    print(f"❌ 测试失败: {fail_count}")
    print(f"❌ 导入错误: {import_error_count}")
    print(f"⏰ 超时: {timeout_count}")
    print(f"❓ 其他: {other_count}")
    print(f"总成功率: {pass_count/len(results)*100:.1f}%")
    
    # 详细分类
    print("\n详细分类:")
    for category, symbol in [('PASS', '✅'), ('FAILED', '❌'), ('IMPORT_ERROR', '🚫'), ('TIMEOUT', '⏰')]:
        files = [name for name, status in results.items() if status == category]
        if files:
            print(f"\n{symbol} {category}:")
            for file in files:
                print(f"  - {file}")

if __name__ == '__main__':
    main()