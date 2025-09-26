#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量运行所有功能测试并生成报告
"""

import subprocess
import os
import sys
from pathlib import Path

def run_test_file(test_file):
    """运行单个测试文件"""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', str(test_file), '-v'],
            capture_output=True,
            text=True,
            cwd=r'E:\project\harborai'
        )
        return {
            'file': test_file.name,
            'success': result.returncode == 0,
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    except Exception as e:
        return {
            'file': test_file.name,
            'success': False,
            'returncode': -1,
            'stdout': '',
            'stderr': str(e)
        }

def main():
    """主函数"""
    functional_dir = Path(r'E:\project\harborai\tests\functional')
    test_files = list(functional_dir.glob('test_*.py'))
    
    # 排除备份文件
    test_files = [f for f in test_files if 'backup' not in f.name]
    
    print(f"找到 {len(test_files)} 个测试文件")
    print("=" * 80)
    
    results = []
    passed_count = 0
    failed_count = 0
    
    for test_file in sorted(test_files):
        print(f"\n运行测试: {test_file.name}")
        print("-" * 40)
        
        result = run_test_file(test_file)
        results.append(result)
        
        if result['success']:
            print(f"✅ {test_file.name} - 通过")
            passed_count += 1
        else:
            print(f"❌ {test_file.name} - 失败 (退出码: {result['returncode']})")
            failed_count += 1
            
            # 显示错误信息的前几行
            if result['stderr']:
                error_lines = result['stderr'].split('\n')[:5]
                print("错误信息:")
                for line in error_lines:
                    if line.strip():
                        print(f"  {line}")
    
    # 生成总结报告
    print("\n" + "=" * 80)
    print("测试总结报告")
    print("=" * 80)
    print(f"总测试文件数: {len(test_files)}")
    print(f"通过: {passed_count}")
    print(f"失败: {failed_count}")
    print(f"成功率: {passed_count/len(test_files)*100:.1f}%")
    
    print("\n通过的测试:")
    for result in results:
        if result['success']:
            print(f"  ✅ {result['file']}")
    
    print("\n失败的测试:")
    for result in results:
        if not result['success']:
            print(f"  ❌ {result['file']} (退出码: {result['returncode']})")
    
    # 保存详细报告到文件
    report_file = r'E:\project\harborai\test_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("功能测试详细报告\n")
        f.write("=" * 80 + "\n\n")
        
        for result in results:
            f.write(f"测试文件: {result['file']}\n")
            f.write(f"状态: {'通过' if result['success'] else '失败'}\n")
            f.write(f"退出码: {result['returncode']}\n")
            f.write("标准输出:\n")
            f.write(result['stdout'])
            f.write("\n标准错误:\n")
            f.write(result['stderr'])
            f.write("\n" + "-" * 80 + "\n\n")
    
    print(f"\n详细报告已保存到: {report_file}")
    
    return failed_count == 0

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)