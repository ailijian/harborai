#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的测试运行器，用于验证测试代码的正确性
"""

import sys
import os
import traceback
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "tests"))

def run_test_module(module_name):
    """运行指定的测试模块"""
    try:
        print(f"\n=== 运行测试模块: {module_name} ===")
        module = __import__(module_name)
        
        # 获取所有测试类
        test_classes = []
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and 
                attr_name.startswith('Test') and 
                attr != type):
                test_classes.append(attr)
        
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        
        for test_class in test_classes:
            print(f"\n--- 测试类: {test_class.__name__} ---")
            
            # 获取所有测试方法
            test_methods = []
            for method_name in dir(test_class):
                if method_name.startswith('test_'):
                    test_methods.append(method_name)
            
            for method_name in test_methods:
                total_tests += 1
                try:
                    # 创建测试实例并运行测试
                    test_instance = test_class()
                    
                    # 运行setUp如果存在
                    if hasattr(test_instance, 'setUp'):
                        test_instance.setUp()
                    
                    # 运行测试方法
                    test_method = getattr(test_instance, method_name)
                    test_method()
                    
                    # 运行tearDown如果存在
                    if hasattr(test_instance, 'tearDown'):
                        test_instance.tearDown()
                    
                    print(f"  ✓ {method_name}")
                    passed_tests += 1
                    
                except Exception as e:
                    print(f"  ✗ {method_name}: {str(e)}")
                    failed_tests += 1
                    # 打印详细错误信息
                    traceback.print_exc()
        
        print(f"\n=== 测试结果 ===")
        print(f"总计: {total_tests}, 通过: {passed_tests}, 失败: {failed_tests}")
        
        return failed_tests == 0
        
    except Exception as e:
        print(f"运行测试模块 {module_name} 时出错: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("HarborAI 测试运行器")
    print("=" * 50)
    
    # 要测试的模块列表
    test_modules = [
        'test_config',
        'test_exceptions', 
        'test_retry',
        'test_integration'
    ]
    
    all_passed = True
    
    for module_name in test_modules:
        try:
            success = run_test_module(module_name)
            if not success:
                all_passed = False
        except ImportError as e:
            print(f"无法导入测试模块 {module_name}: {str(e)}")
            all_passed = False
        except Exception as e:
            print(f"运行测试模块 {module_name} 时出现未知错误: {str(e)}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 所有测试通过！")
        sys.exit(0)
    else:
        print("❌ 部分测试失败")
        sys.exit(1)

if __name__ == "__main__":
    main()