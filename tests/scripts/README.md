# HarborAI 测试脚本使用说明

本目录包含了 HarborAI 项目的测试自动化脚本，适用于 Windows 11 + PowerShell 环境。

## 脚本概览

| 脚本名称 | 功能描述 | 主要用途 |
|---------|---------|----------|
| `setup_test_env.ps1` | 测试环境设置 | 初始化测试环境，安装依赖，配置数据库 |
| `run_all_tests.ps1` | 运行所有测试 | 执行功能测试、集成测试等 |
| `run_performance_tests.ps1` | 运行性能测试 | 执行负载测试、压力测试、基准测试 |
| `generate_reports.ps1` | 生成测试报告 | 生成各种格式的测试报告 |
| `cleanup_test_env.ps1` | 清理测试环境 | 清理临时文件、容器、数据库等 |

## 快速开始

### 1. 环境准备

确保系统已安装以下软件：
- Python 3.8+ 
- PowerShell 5.1+
- Docker Desktop（可选）
- Git

### 2. 初始化测试环境

```powershell
# 基本设置
.\setup_test_env.ps1

# 跳过 Docker 设置
.\setup_test_env.ps1 -SkipDocker

# 跳过数据库设置
.\setup_test_env.ps1 -SkipDatabase

# 指定 Python 版本
.\setup_test_env.ps1 -PythonVersion "3.9"
```

### 3. 运行测试

```powershell
# 运行所有测试
.\run_all_tests.ps1

# 运行特定类型的测试
.\run_all_tests.ps1 -TestType unit
.\run_all_tests.ps1 -TestType integration
.\run_all_tests.ps1 -TestType security

# 运行高优先级测试
.\run_all_tests.ps1 -Priority p0

# 生成覆盖率报告
.\run_all_tests.ps1 -Coverage -GenerateReport
```

### 4. 性能测试

```powershell
# 基本性能测试
.\run_performance_tests.ps1

# 指定并发数和持续时间
.\run_performance_tests.ps1 -TestType load -Concurrency 50 -Duration 300

# 运行压力测试
.\run_performance_tests.ps1 -TestType stress -Concurrency 100 -Duration 600

# 运行基准测试
.\run_performance_tests.ps1 -TestType benchmark
```

### 5. 生成报告

```powershell
# 生成所有报告
.\generate_reports.ps1

# 生成特定类型报告
.\generate_reports.ps1 -ReportType summary
.\generate_reports.ps1 -ReportType coverage
.\generate_reports.ps1 -ReportType performance

# 指定输出格式
.\generate_reports.ps1 -ReportFormat html
.\generate_reports.ps1 -ReportFormat pdf
```

### 6. 清理环境

```powershell
# 完整清理
.\cleanup_test_env.ps1

# 仅清理容器
.\cleanup_test_env.ps1 -CleanupType containers

# 强制清理（无确认提示）
.\cleanup_test_env.ps1 -Force

# 保留测试报告
.\cleanup_test_env.ps1 -KeepReports
```

## 详细使用说明

### setup_test_env.ps1

**功能**：初始化和配置 HarborAI 测试环境

**参数**：
- `-SkipDocker`：跳过 Docker 环境设置
- `-SkipDatabase`：跳过数据库初始化
- `-PythonVersion`：指定 Python 版本（默认 3.9）
- `-Verbose`：显示详细输出
- `-Help`：显示帮助信息

**执行步骤**：
1. 检查 Python 环境
2. 创建并激活虚拟环境
3. 安装测试依赖
4. 设置环境变量
5. 启动 Docker 服务（可选）
6. 初始化测试数据库（可选）
7. 创建报告目录
8. 验证环境配置

**示例**：
```powershell
# 完整环境设置
.\setup_test_env.ps1 -Verbose

# 仅设置 Python 环境
.\setup_test_env.ps1 -SkipDocker -SkipDatabase
```

### run_all_tests.ps1

**功能**：执行各种类型的测试

**参数**：
- `-TestType`：测试类型（all, unit, integration, api, security, compatibility）
- `-Priority`：测试优先级（all, p0, p1, p2, p3）
- `-Coverage`：生成代码覆盖率报告
- `-Parallel`：并行执行测试
- `-FailFast`：遇到失败立即停止
- `-ReportFormat`：报告格式（html, xml, json, allure）
- `-Verbose`：显示详细输出
- `-Help`：显示帮助信息

**测试类型说明**：
- `unit`：单元测试
- `integration`：集成测试
- `api`：API 测试
- `security`：安全测试
- `compatibility`：兼容性测试
- `all`：所有测试

**示例**：
```powershell
# 运行所有 P0 优先级测试
.\run_all_tests.ps1 -Priority p0 -Coverage -Verbose

# 并行运行单元测试
.\run_all_tests.ps1 -TestType unit -Parallel

# 生成 Allure 报告
.\run_all_tests.ps1 -ReportFormat allure
```

### run_performance_tests.ps1

**功能**：执行性能测试和基准测试

**参数**：
- `-TestType`：测试类型（load, stress, memory, response, throughput, benchmark）
- `-Duration`：测试持续时间（秒）
- `-Concurrency`：并发用户数
- `-RampUp`：压力递增时间（秒）
- `-Endpoint`：测试端点 URL
- `-ReportFormat`：报告格式（html, json, csv）
- `-Verbose`：显示详细输出
- `-Help`：显示帮助信息

**测试类型说明**：
- `load`：负载测试
- `stress`：压力测试
- `memory`：内存测试
- `response`：响应时间测试
- `throughput`：吞吐量测试
- `benchmark`：基准测试

**示例**：
```powershell
# 负载测试：50 并发用户，持续 5 分钟
.\run_performance_tests.ps1 -TestType load -Concurrency 50 -Duration 300

# 压力测试：逐步增加到 100 并发
.\run_performance_tests.ps1 -TestType stress -Concurrency 100 -RampUp 60

# 基准测试
.\run_performance_tests.ps1 -TestType benchmark -ReportFormat json
```

### generate_reports.ps1

**功能**：生成各种格式的测试报告

**参数**：
- `-ReportType`：报告类型（summary, coverage, performance, security, trend, dashboard, all）
- `-ReportFormat`：报告格式（html, pdf, json, xml, csv）
- `-OutputDir`：输出目录
- `-IncludeCharts`：包含图表
- `-EmailNotification`：发送邮件通知
- `-Verbose`：显示详细输出
- `-Help`：显示帮助信息

**报告类型说明**：
- `summary`：测试摘要报告
- `coverage`：代码覆盖率报告
- `performance`：性能测试报告
- `security`：安全测试报告
- `trend`：趋势分析报告
- `dashboard`：仪表板报告
- `all`：所有报告

**示例**：
```powershell
# 生成 HTML 格式的摘要报告
.\generate_reports.ps1 -ReportType summary -ReportFormat html

# 生成包含图表的性能报告
.\generate_reports.ps1 -ReportType performance -IncludeCharts

# 生成所有报告并发送邮件
.\generate_reports.ps1 -ReportType all -EmailNotification
```

### cleanup_test_env.ps1

**功能**：清理测试环境和临时文件

**参数**：
- `-CleanupType`：清理类型（all, containers, database, files, reports, env, python）
- `-Force`：强制清理（无确认提示）
- `-KeepLogs`：保留日志文件
- `-KeepReports`：保留测试报告
- `-Verbose`：显示详细输出
- `-Help`：显示帮助信息

**清理类型说明**：
- `all`：完整清理
- `containers`：清理 Docker 容器
- `database`：清理测试数据库
- `files`：清理临时文件
- `reports`：清理测试报告
- `env`：重置环境变量
- `python`：清理 Python 虚拟环境

**示例**：
```powershell
# 完整清理但保留报告
.\cleanup_test_env.ps1 -KeepReports

# 仅清理 Docker 容器
.\cleanup_test_env.ps1 -CleanupType containers -Force

# 清理临时文件但保留日志
.\cleanup_test_env.ps1 -CleanupType files -KeepLogs
```

## 常见问题

### Q: 脚本执行失败，提示权限不足
**A**: 以管理员身份运行 PowerShell，或执行以下命令：
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Q: Docker 服务启动失败
**A**: 确保 Docker Desktop 已安装并正在运行，或使用 `-SkipDocker` 参数跳过 Docker 设置。

### Q: 测试执行时间过长
**A**: 使用 `-Priority p0` 仅运行高优先级测试，或使用 `-TestType unit` 仅运行单元测试。

### Q: 报告生成失败
**A**: 检查输出目录权限，确保有足够的磁盘空间，或使用 `-Verbose` 查看详细错误信息。

### Q: 环境变量设置不生效
**A**: 重新启动 PowerShell 会话，或手动设置环境变量：
```powershell
$env:VARIABLE_NAME = "value"
```

## 最佳实践

1. **定期清理**：定期运行清理脚本，避免磁盘空间不足
2. **增量测试**：开发过程中使用 `-Priority p0` 快速验证
3. **完整测试**：发布前运行完整测试套件
4. **报告归档**：重要的测试报告应及时归档
5. **环境隔离**：使用虚拟环境避免依赖冲突

## 技术支持

如遇到问题，请：
1. 查看脚本日志文件（`tests/logs/` 目录）
2. 使用 `-Verbose` 参数获取详细信息
3. 检查环境配置和依赖安装
4. 参考技术设计方案文档

---

**注意**：所有脚本都遵循 VIBE Coding 规则，包含详细的中文注释和错误处理机制。