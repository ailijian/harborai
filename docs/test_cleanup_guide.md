# HarborAI 测试结果文件清理指南

## 概述

本指南介绍如何使用 HarborAI 项目的测试结果文件清理脚本，用于清理项目内所有测试报告、缓存文件和临时结果文件。

## 清理脚本

项目提供了两个清理脚本：

1. **Python 脚本** (推荐): `tests/scripts/cleanup_test_results.py`
2. **PowerShell 脚本**: `tests/scripts/cleanup_all_test_results.ps1`

## 清理范围

脚本会清理以下文件和目录：

### 测试报告目录
- `tests/reports/` - 所有测试报告文件
- `tests/performance/performance_reports/` - 性能测试报告
- `tests/performance/htmlcov_*` - HTML 覆盖率报告目录
- `tests/performance/metrics/` - 性能指标文件
- `tests/performance/test_results/` - 测试结果文件
- `reports/` - 根目录报告文件夹

### 缓存和临时文件
- `__pycache__/` - Python 字节码缓存目录（递归查找）
- `.pytest_cache/` - pytest 缓存目录（递归查找）
- `.coverage` - 覆盖率数据文件
- `*.coverage` - 覆盖率相关文件
- `htmlcov/` - HTML 覆盖率报告目录
- `.benchmarks/` - 基准测试缓存
- `metrics/` - 指标文件目录

### 日志文件
- `tests/logs/` - 测试日志文件
- `tests/performance/*.log` - 性能测试日志
- `tests/reports/*.log` - 报告日志文件

### 保留项目
脚本不会删除以下文件：
- 测试源代码文件 (`.py`, `.js`, `.ts` 等)
- 配置文件 (`pytest.ini`, `.env.test`, `pyproject.toml` 等)
- 文档文件 (`README.md`, `*.md` 文档)
- 脚本文件本身

## 使用方法

### Python 脚本 (推荐)

#### 基本用法

```bash
# 预览将要删除的文件（干运行模式）
python tests/scripts/cleanup_test_results.py --dry-run

# 创建备份并清理
python tests/scripts/cleanup_test_results.py --backup

# 强制清理（跳过确认）
python tests/scripts/cleanup_test_results.py --force

# 指定项目根目录
python tests/scripts/cleanup_test_results.py --project-root /path/to/project

# 自定义备份路径
python tests/scripts/cleanup_test_results.py --backup --backup-path /path/to/backup
```

#### 参数说明

- `--dry-run`: 干运行模式，仅显示将要删除的文件，不执行实际删除
- `--backup`: 在删除前创建备份
- `--force`: 强制删除，跳过确认提示
- `--project-root`: 指定项目根目录路径（默认为当前目录）
- `--backup-path`: 自定义备份路径

### PowerShell 脚本

#### 基本用法

```powershell
# 预览将要删除的文件（干运行模式）
.\tests\scripts\cleanup_all_test_results.ps1 -DryRun

# 创建备份并清理
.\tests\scripts\cleanup_all_test_results.ps1 -Backup

# 强制清理（跳过确认）
.\tests\scripts\cleanup_all_test_results.ps1 -Force

# 指定项目根目录
.\tests\scripts\cleanup_all_test_results.ps1 -ProjectRoot "C:\path\to\project"

# 自定义备份路径
.\tests\scripts\cleanup_all_test_results.ps1 -Backup -BackupPath "C:\path\to\backup"
```

#### 参数说明

- `-DryRun`: 干运行模式，仅显示将要删除的文件
- `-Backup`: 在删除前创建备份
- `-Force`: 强制删除，跳过确认提示
- `-ProjectRoot`: 指定项目根目录路径
- `-BackupPath`: 自定义备份路径

## 使用示例

### 场景 1: 首次使用，预览清理内容

```bash
# 使用 Python 脚本预览
python tests/scripts/cleanup_test_results.py --dry-run
```

输出示例：
```
[2025-01-05 10:52:42,123] [INFO] 扫描测试结果文件和目录
[2025-01-05 10:52:42,145] [INFO] 发现目录: E:\project\harborai\tests\reports
[2025-01-05 10:52:42,156] [INFO] 发现 __pycache__ 目录: E:\project\harborai\harborai\__pycache__
...

将要删除的文件:
  - E:\project\harborai\.coverage (1.23 KB)

将要删除的目录:
  - E:\project\harborai\tests\reports (2.45 MB)
  - E:\project\harborai\harborai\__pycache__ (1.03 KB)
  ...

总计: 19 个文件，46 个目录，28.56 MB
```

### 场景 2: 安全清理（推荐）

```bash
# 创建备份并清理
python tests/scripts/cleanup_test_results.py --backup
```

这会：
1. 扫描所有测试结果文件
2. 创建备份到 `backup_test_results_YYYYMMDD_HHMMSS` 目录
3. 显示确认提示
4. 执行清理操作
5. 生成清理报告

### 场景 3: 快速清理（CI/CD 环境）

```bash
# 强制清理，跳过确认
python tests/scripts/cleanup_test_results.py --force
```

### 场景 4: 清理特定项目

```bash
# 清理指定项目目录
python tests/scripts/cleanup_test_results.py --project-root /path/to/another/project --backup
```

## 清理报告

清理完成后，脚本会生成详细的清理报告：

### 报告位置
- 日志文件: `logs/cleanup_test_results_YYYYMMDD_HHMMSS.log`
- 清理报告: `logs/cleanup_report_YYYYMMDD_HHMMSS.md`

### 报告内容
- 清理摘要（开始/结束时间、持续时间、删除统计）
- 删除的文件列表
- 删除的目录列表
- 错误信息（如有）
- 备份信息

### 示例报告

```markdown
# HarborAI 测试结果清理报告

## 清理摘要
- **开始时间**: 2025-01-05 10:52:42
- **结束时间**: 2025-01-05 10:52:45
- **持续时间**: 00:00:03
- **删除文件数**: 19
- **删除目录数**: 46
- **释放空间**: 28.56 MB
- **错误数量**: 0

## 删除的文件
- E:\project\harborai\.coverage
- E:\project\harborai\tests\performance\test_results.json
...

## 删除的目录
- E:\project\harborai\tests\reports
- E:\project\harborai\harborai\__pycache__
...

## 备份信息
备份路径: E:\project\harborai\backup_test_results_20250105_105242
```

## 安全特性

### 1. 干运行模式
- 使用 `--dry-run` 参数预览将要删除的内容
- 不执行实际删除操作
- 显示文件大小和总计信息

### 2. 备份机制
- 使用 `--backup` 参数创建删除前备份
- 备份包含所有将要删除的文件和目录
- 备份路径包含时间戳，避免冲突

### 3. 确认机制
- 默认显示删除确认提示
- 显示删除项目数量和总大小
- 用户可以取消操作

### 4. 详细日志
- 记录所有操作到日志文件
- 包含时间戳和操作级别
- 错误信息详细记录

### 5. 错误处理
- 单个文件删除失败不影响其他操作
- 详细记录错误信息
- 继续执行剩余清理任务

## 故障排除

### 常见问题

#### 1. 权限错误
```
错误: 删除文件失败 /path/to/file: Permission denied
```

**解决方案**:
- 确保有足够的文件系统权限
- 在 Windows 上以管理员身份运行
- 检查文件是否被其他进程占用

#### 2. 路径不存在
```
错误: 未找到 tests 目录，请确认在正确的项目根目录下运行脚本
```

**解决方案**:
- 确认在 HarborAI 项目根目录下运行
- 使用 `--project-root` 参数指定正确路径

#### 3. 编码问题（PowerShell）
```
字符串缺少终止符
```

**解决方案**:
- 使用 Python 脚本替代 PowerShell 脚本
- 确保 PowerShell 使用 UTF-8 编码

### 恢复数据

如果意外删除了重要文件：

1. **从备份恢复**（如果使用了 `--backup`）:
   ```bash
   # 复制备份文件回原位置
   cp -r backup_test_results_YYYYMMDD_HHMMSS/* .
   ```

2. **从版本控制恢复**:
   ```bash
   # 恢复被删除的文件
   git checkout HEAD -- tests/
   ```

## 最佳实践

### 1. 定期清理
- 建议在每次重要测试运行后清理
- CI/CD 流水线中集成自动清理
- 开发环境定期手动清理

### 2. 使用备份
- 首次使用时总是创建备份
- 重要环境中使用备份选项
- 定期清理旧备份文件

### 3. 预览优先
- 使用 `--dry-run` 预览清理内容
- 确认清理范围符合预期
- 避免误删重要文件

### 4. 监控空间
- 清理前后对比磁盘空间
- 关注清理报告中的空间释放信息
- 定期监控测试文件增长

## 集成到 CI/CD

### GitHub Actions 示例

```yaml
name: Cleanup Test Results
on:
  workflow_run:
    workflows: ["Test Suite"]
    types: [completed]

jobs:
  cleanup:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Cleanup test results
        run: |
          python tests/scripts/cleanup_test_results.py --force
```

### Jenkins 示例

```groovy
pipeline {
    agent any
    stages {
        stage('Cleanup') {
            steps {
                script {
                    sh 'python tests/scripts/cleanup_test_results.py --force'
                }
            }
        }
    }
    post {
        always {
            archiveArtifacts artifacts: 'logs/cleanup_report_*.md', allowEmptyArchive: true
        }
    }
}
```

## 总结

HarborAI 测试结果清理脚本提供了安全、可靠的方式来清理项目中的测试文件和缓存。通过合理使用干运行模式、备份机制和详细日志，可以确保清理操作的安全性和可追溯性。

建议优先使用 Python 脚本，因为它具有更好的跨平台兼容性和稳定性。在生产环境中使用时，请务必先进行预览和备份。