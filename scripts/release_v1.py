#!/usr/bin/env python3
"""
---
summary: HarborAI 1.0.0 正式版发布脚本
description: 将版本从beta更新为正式版并触发PyPI发布流程
author: HarborAI Team
version: 1.0.0
---

HarborAI 1.0.0 正式版发布自动化脚本

功能：
1. 更新 pyproject.toml 版本号从 beta 到正式版
2. 更新开发状态为 Production/Stable
3. 创建发布 commit 和标签
4. 推送到远程仓库触发 GitHub Actions 发布流程

使用方法：
    python scripts/release_v1.py

安全检查：
- 工作目录必须干净（无未提交更改）
- 当前分支必须是 main 或 master
- 远程仓库连接正常
"""

import os
import sys
import subprocess
import re
from pathlib import Path
from typing import Tuple, Optional


class ReleaseManager:
    """
    ---
    summary: 发布管理器
    description: 负责版本更新、Git操作和发布流程管理
    ---
    """
    
    def __init__(self):
        """初始化发布管理器"""
        self.project_root = Path(__file__).parent.parent
        self.pyproject_path = self.project_root / "pyproject.toml"
        self.target_version = "1.0.0"
        self.target_tag = f"v{self.target_version}"
        
    def print_step(self, step: str, message: str) -> None:
        """
        打印步骤信息
        
        Args:
            step: 步骤标识
            message: 步骤描述
        """
        print(f"\n🔄 [{step}] {message}")
        
    def print_success(self, message: str) -> None:
        """打印成功信息"""
        print(f"✅ {message}")
        
    def print_error(self, message: str) -> None:
        """打印错误信息"""
        print(f"❌ {message}")
        
    def print_warning(self, message: str) -> None:
        """打印警告信息"""
        print(f"⚠️ {message}")
        
    def confirm_action(self, message: str) -> bool:
        """
        请求用户确认操作
        
        Args:
            message: 确认信息
            
        Returns:
            用户确认结果
        """
        while True:
            response = input(f"\n❓ {message} (y/n): ").lower().strip()
            if response in ['y', 'yes', '是']:
                return True
            elif response in ['n', 'no', '否']:
                return False
            else:
                print("请输入 y/yes/是 或 n/no/否")
    
    def run_command(self, cmd: str, capture_output: bool = True) -> Tuple[bool, str]:
        """
        执行命令
        
        Args:
            cmd: 要执行的命令
            capture_output: 是否捕获输出
            
        Returns:
            (成功状态, 输出内容)
        """
        try:
            if capture_output:
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, cwd=self.project_root
                )
                return result.returncode == 0, result.stdout.strip()
            else:
                result = subprocess.run(cmd, shell=True, cwd=self.project_root)
                return result.returncode == 0, ""
        except Exception as e:
            return False, str(e)
    
    def check_git_status(self) -> bool:
        """
        检查Git工作目录状态
        
        Returns:
            工作目录是否干净
        """
        self.print_step("1", "检查Git工作目录状态")
        
        success, output = self.run_command("git status --porcelain")
        if not success:
            self.print_error("无法获取Git状态")
            return False
            
        if output.strip():
            self.print_error("工作目录不干净，存在未提交的更改：")
            print(output)
            return False
            
        self.print_success("工作目录干净")
        return True
    
    def check_current_branch(self) -> bool:
        """
        检查当前分支
        
        Returns:
            是否在主分支
        """
        self.print_step("2", "检查当前分支")
        
        success, branch = self.run_command("git branch --show-current")
        if not success:
            self.print_error("无法获取当前分支")
            return False
            
        if branch not in ['main', 'master']:
            self.print_error(f"当前分支是 '{branch}'，请切换到 main 或 master 分支")
            return False
            
        self.print_success(f"当前分支: {branch}")
        return True
    
    def check_remote_connection(self) -> bool:
        """
        检查远程仓库连接
        
        Returns:
            远程连接是否正常
        """
        self.print_step("3", "检查远程仓库连接")
        
        success, _ = self.run_command("git fetch --dry-run")
        if not success:
            self.print_error("无法连接到远程仓库")
            return False
            
        self.print_success("远程仓库连接正常")
        return True
    
    def update_version_in_pyproject(self) -> bool:
        """
        更新 pyproject.toml 中的版本号和开发状态
        
        Returns:
            更新是否成功
        """
        self.print_step("4", f"更新版本号到 {self.target_version}")
        
        if not self.pyproject_path.exists():
            self.print_error(f"找不到 pyproject.toml 文件: {self.pyproject_path}")
            return False
        
        try:
            # 读取文件内容
            content = self.pyproject_path.read_text(encoding='utf-8')
            
            # 更新版本号
            version_pattern = r'version\s*=\s*"[^"]*"'
            new_version_line = f'version = "{self.target_version}"'
            content = re.sub(version_pattern, new_version_line, content)
            
            # 更新开发状态
            status_pattern = r'"Development Status :: 4 - Beta"'
            new_status = '"Development Status :: 5 - Production/Stable"'
            content = re.sub(status_pattern, new_status, content)
            
            # 写回文件
            self.pyproject_path.write_text(content, encoding='utf-8')
            
            self.print_success(f"版本号已更新为 {self.target_version}")
            self.print_success("开发状态已更新为 Production/Stable")
            return True
            
        except Exception as e:
            self.print_error(f"更新 pyproject.toml 失败: {e}")
            return False
    
    def create_release_commit(self) -> bool:
        """
        创建发布提交
        
        Returns:
            提交是否成功
        """
        self.print_step("5", "创建发布提交")
        
        # 添加更改
        success, _ = self.run_command("git add pyproject.toml")
        if not success:
            self.print_error("添加文件到暂存区失败")
            return False
        
        # 创建提交
        commit_message = f"🚀 Release v{self.target_version}\n\n- 更新版本号到 {self.target_version}\n- 更新开发状态为 Production/Stable"
        success, _ = self.run_command(f'git commit -m "{commit_message}"')
        if not success:
            self.print_error("创建提交失败")
            return False
            
        self.print_success("发布提交创建成功")
        return True
    
    def create_release_tag(self) -> bool:
        """
        创建发布标签
        
        Returns:
            标签创建是否成功
        """
        self.print_step("6", f"创建发布标签 {self.target_tag}")
        
        tag_message = f"HarborAI {self.target_version} 正式版发布"
        success, _ = self.run_command(f'git tag -a {self.target_tag} -m "{tag_message}"')
        if not success:
            self.print_error("创建标签失败")
            return False
            
        self.print_success(f"标签 {self.target_tag} 创建成功")
        return True
    
    def push_to_remote(self) -> bool:
        """
        推送到远程仓库
        
        Returns:
            推送是否成功
        """
        self.print_step("7", "推送到远程仓库")
        
        # 推送提交
        success, _ = self.run_command("git push")
        if not success:
            self.print_error("推送提交失败")
            return False
        
        # 推送标签
        success, _ = self.run_command(f"git push origin {self.target_tag}")
        if not success:
            self.print_error("推送标签失败")
            return False
            
        self.print_success("推送到远程仓库成功")
        return True
    
    def show_release_info(self) -> None:
        """显示发布信息"""
        print("\n" + "="*60)
        print("🎉 HarborAI 1.0.0 正式版发布流程已启动！")
        print("="*60)
        print(f"📦 版本号: {self.target_version}")
        print(f"🏷️ Git标签: {self.target_tag}")
        print(f"🚀 GitHub Actions: 发布流程已自动触发")
        print(f"📋 发布页面: https://github.com/harborai/harborai/releases/tag/{self.target_tag}")
        print(f"📦 PyPI页面: https://pypi.org/project/harborai/")
        print("\n⏳ 请等待 GitHub Actions 完成构建和发布...")
        print("📧 发布完成后会收到通知")
        print("="*60)
    
    def rollback_changes(self) -> None:
        """回滚更改"""
        self.print_warning("正在回滚更改...")
        
        # 重置文件更改
        self.run_command("git checkout -- pyproject.toml")
        
        # 删除可能创建的标签
        self.run_command(f"git tag -d {self.target_tag}")
        
        # 重置最后一次提交（如果存在）
        success, output = self.run_command("git log --oneline -1")
        if success and f"Release v{self.target_version}" in output:
            self.run_command("git reset --hard HEAD~1")
        
        self.print_success("更改已回滚")
    
    def run_release(self) -> bool:
        """
        执行完整的发布流程
        
        Returns:
            发布是否成功
        """
        print("🚀 HarborAI 1.0.0 正式版发布脚本")
        print("="*50)
        
        try:
            # 安全检查
            if not self.check_git_status():
                return False
            
            if not self.check_current_branch():
                return False
            
            if not self.check_remote_connection():
                return False
            
            # 确认发布
            if not self.confirm_action(f"确认发布 HarborAI {self.target_version} 正式版？"):
                print("发布已取消")
                return False
            
            # 执行发布步骤
            if not self.update_version_in_pyproject():
                return False
            
            if not self.create_release_commit():
                return False
            
            if not self.create_release_tag():
                return False
            
            # 最终确认推送
            if not self.confirm_action("确认推送到远程仓库并触发发布流程？"):
                self.rollback_changes()
                print("发布已取消，更改已回滚")
                return False
            
            if not self.push_to_remote():
                return False
            
            # 显示发布信息
            self.show_release_info()
            return True
            
        except KeyboardInterrupt:
            print("\n\n⚠️ 发布被用户中断")
            if self.confirm_action("是否回滚已做的更改？"):
                self.rollback_changes()
            return False
        except Exception as e:
            self.print_error(f"发布过程中出现错误: {e}")
            if self.confirm_action("是否回滚已做的更改？"):
                self.rollback_changes()
            return False


def main():
    """主函数"""
    release_manager = ReleaseManager()
    
    success = release_manager.run_release()
    
    if success:
        print("\n✅ 发布流程启动成功！")
        sys.exit(0)
    else:
        print("\n❌ 发布流程失败！")
        sys.exit(1)


if __name__ == "__main__":
    main()