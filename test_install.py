#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试从 TestPyPI 安装 HarborAI 预发布版本的脚本

使用方法：
1. 创建新的虚拟环境
2. 运行此脚本进行安装和验证
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"\n🔄 {description}")
    print(f"执行命令: {cmd}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(f"返回码: {result.returncode}")
        if result.stdout:
            print(f"输出:\n{result.stdout}")
        if result.stderr:
            print(f"错误:\n{result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"执行失败: {e}")
        return False

def main():
    """主测试流程"""
    print("🚀 HarborAI v1.0.0-beta.1 TestPyPI 安装测试")
    print("=" * 60)
    
    # 1. 安装预发布版本
    install_cmd = "pip install -i https://test.pypi.org/simple/ harborai==1.0.0-beta.1"
    if not run_command(install_cmd, "从 TestPyPI 安装 HarborAI"):
        print("❌ 安装失败")
        return False
    
    # 2. 验证安装
    verify_cmd = "python -c \"import harborai; print(f'HarborAI 版本: {harborai.__version__}')\""
    if not run_command(verify_cmd, "验证安装和版本"):
        print("❌ 验证失败")
        return False
    
    # 3. 测试基本导入
    test_cmd = "python -c \"from harborai import HarborAI; print('✅ HarborAI 导入成功')\""
    if not run_command(test_cmd, "测试基本导入"):
        print("❌ 导入测试失败")
        return False
    
    print("\n🎉 所有测试通过！HarborAI v1.0.0-beta.1 安装成功")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)