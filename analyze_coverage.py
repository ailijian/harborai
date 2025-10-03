#!/usr/bin/env python3
"""
性能测试框架覆盖率分析脚本
分析当前测试覆盖率并生成报告
"""

import os
import ast
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

def analyze_module_coverage(module_path: str, test_files: List[str]) -> Dict:
    """分析模块的测试覆盖率"""
    
    # 解析模块文件
    with open(module_path, 'r', encoding='utf-8') as f:
        module_content = f.read()
    
    try:
        module_tree = ast.parse(module_content)
    except SyntaxError as e:
        return {"error": f"语法错误: {e}"}
    
    # 提取模块中的类和函数
    classes = []
    functions = []
    
    for node in ast.walk(module_tree):
        if isinstance(node, ast.ClassDef):
            classes.append(node.name)
            # 提取类中的方法
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    functions.append(f"{node.name}.{item.name}")
        elif isinstance(node, ast.FunctionDef) and not any(isinstance(parent, ast.ClassDef) for parent in ast.walk(module_tree) if hasattr(parent, 'body') and node in getattr(parent, 'body', [])):
            functions.append(item.name)
    
    # 分析测试文件中的覆盖情况
    tested_items = set()
    
    for test_file in test_files:
        if os.path.exists(test_file):
            with open(test_file, 'r', encoding='utf-8') as f:
                test_content = f.read()
            
            # 简单的字符串匹配来检测测试覆盖
            for class_name in classes:
                if class_name in test_content:
                    tested_items.add(class_name)
            
            for func_name in functions:
                if func_name in test_content or func_name.split('.')[-1] in test_content:
                    tested_items.add(func_name)
    
    total_items = len(classes) + len(functions)
    tested_count = len(tested_items)
    coverage_percentage = (tested_count / total_items * 100) if total_items > 0 else 0
    
    return {
        "module": os.path.basename(module_path),
        "classes": classes,
        "functions": functions,
        "tested_items": list(tested_items),
        "total_items": total_items,
        "tested_count": tested_count,
        "coverage_percentage": coverage_percentage,
        "untested_items": list(set(classes + functions) - tested_items)
    }

def main():
    """主函数"""
    performance_dir = Path("tests/performance")
    
    # 核心模块
    core_modules = [
        "tests/performance/core_performance_framework.py",
        "tests/performance/performance_report_generator.py", 
        "tests/performance/performance_test_controller.py"
    ]
    
    # 测试文件
    test_files = [
        "tests/performance/test_core_performance_framework.py",
        "tests/performance/test_performance_report_generator.py",
        "tests/performance/test_performance_test_controller.py",
        "tests/performance/test_performance_test_controller_simple.py"
    ]
    
    print("=== 性能测试框架覆盖率分析报告 ===\n")
    
    total_coverage = 0
    module_count = 0
    
    for module_path in core_modules:
        if os.path.exists(module_path):
            print(f"分析模块: {module_path}")
            result = analyze_module_coverage(module_path, test_files)
            
            if "error" in result:
                print(f"  错误: {result['error']}\n")
                continue
            
            print(f"  类: {len(result['classes'])}")
            print(f"  函数: {len(result['functions'])}")
            print(f"  总项目: {result['total_items']}")
            print(f"  已测试: {result['tested_count']}")
            print(f"  覆盖率: {result['coverage_percentage']:.1f}%")
            
            if result['untested_items']:
                print(f"  未测试项目: {', '.join(result['untested_items'][:5])}")
                if len(result['untested_items']) > 5:
                    print(f"    ... 还有 {len(result['untested_items']) - 5} 个")
            
            print()
            
            total_coverage += result['coverage_percentage']
            module_count += 1
        else:
            print(f"模块不存在: {module_path}\n")
    
    if module_count > 0:
        avg_coverage = total_coverage / module_count
        print(f"=== 总体覆盖率: {avg_coverage:.1f}% ===")
        
        if avg_coverage < 80:
            print("⚠️  覆盖率低于80%目标，需要补充测试用例")
        elif avg_coverage < 90:
            print("✅ 覆盖率达到80%目标，但未达到90%理想目标")
        else:
            print("🎉 覆盖率达到90%理想目标！")
    
    # 检查测试文件数量
    print(f"\n=== 测试文件统计 ===")
    existing_tests = [f for f in test_files if os.path.exists(f)]
    print(f"存在的测试文件: {len(existing_tests)}/{len(test_files)}")
    for test_file in existing_tests:
        size = os.path.getsize(test_file)
        print(f"  {test_file}: {size} bytes")

if __name__ == "__main__":
    main()