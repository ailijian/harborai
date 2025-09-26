#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单独运行test_env_file_loading测试
"""

import sys
import os
import tempfile
sys.path.append('.')

from tests.functional.test_k_configuration import TestEnvironmentVariables, MockEnvironmentManager

def run_single_test():
    """运行单个测试方法"""
    print("开始运行test_env_file_loading测试...")
    
    test_instance = TestEnvironmentVariables()
    
    try:
        test_instance.test_env_file_loading()
        print("✓ 测试通过!")
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_single_test()
    sys.exit(0 if success else 1)