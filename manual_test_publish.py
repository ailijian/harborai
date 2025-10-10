#!/usr/bin/env python3
"""
手动测试 TestPyPI 发布脚本

用于验证包构建和发布流程是否正常工作。
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """运行命令并处理错误"""
    print(f"\n🔄 {description}")
    print(f"执行命令: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} 成功")
        if result.stdout:
            print(f"输出: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失败")
        print(f"错误: {e.stderr}")
        return False

def main():
    """主函数"""
    print("🚀 HarborAI TestPyPI 手动发布测试")
    print("=" * 50)
    
    # 检查当前目录
    if not Path("pyproject.toml").exists():
        print("❌ 请在项目根目录运行此脚本")
        sys.exit(1)
    
    # 清理旧的构建文件
    print("\n🧹 清理旧的构建文件")
    for path in ["dist", "build", "*.egg-info"]:
        if os.path.exists(path):
            if os.path.isdir(path):
                import shutil
                shutil.rmtree(path)
                print(f"删除目录: {path}")
            else:
                os.remove(path)
                print(f"删除文件: {path}")
    
    # 安装构建工具
    if not run_command("pip install --upgrade build twine", "安装构建工具"):
        sys.exit(1)
    
    # 构建包
    if not run_command("python -m build", "构建 Python 包"):
        sys.exit(1)
    
    # 检查包
    if not run_command("python -m twine check dist/*", "检查包完整性"):
        sys.exit(1)
    
    print("\n✅ 包构建和检查完成！")
    print("\n📦 构建的文件:")
    for file in Path("dist").glob("*"):
        print(f"  - {file}")
    
    print("\n🔑 要发布到 TestPyPI，您需要：")
    print("1. 在 TestPyPI 上创建 harborai 项目")
    print("2. 配置 Trusted Publishing")
    print("3. 或者使用 API Token:")
    print("   python -m twine upload --repository testpypi dist/*")
    
    print("\n🎯 GitHub Actions 工作流将自动处理发布")

if __name__ == "__main__