<#
.SYNOPSIS
    HarborAI测试报告迁移脚本

.DESCRIPTION
    此脚本将现有分散的测试报告迁移到统一的目录结构中，包括：
    - 覆盖率报告迁移
    - HTML测试报告迁移
    - Allure报告迁移
    - 性能测试报告迁移
    - 安全测试报告迁移
    - 日志文件迁移
    - 创建备份和回滚点

.PARAMETER SourceDir
    源报告目录（默认：项目根目录下的reports目录）

.PARAMETER TargetDir
    目标报告目录（默认：tests/reports）

.PARAMETER BackupDir
    备份目录（默认：tests/backups/reports_backup_YYYYMMDD_HHMMSS）

.PARAMETER DryRun
    仅显示将要执行的操作，不实际移动文件

.PARAMETER Force
    强制迁移，覆盖已存在的文件

.PARAMETER SkipBackup
    跳过备份创建

.PARAMETER Verbose
    显示详细输出

.PARAMETER Help
    显示帮助信息

.EXAMPLE
    .\migrate_reports.ps1
    使用默认设置迁移报告

.EXAMPLE
    .\migrate_reports.ps1 -DryRun -Verbose
    预览迁移操作，显示详细信息

.EXAMPLE
    .\migrate_reports.ps1 -Force -SkipBackup
    强制迁移，不创建备份
#>

param(
    [string]$SourceDir = "",
    [string]$TargetDir = "",
    [string]$BackupDir = "",
    [switch]$DryRun,
    [switch]$Force,
    [switch]$SkipBackup,
    [switch]$Verbose,
    [switch]$Help
)

# 显示帮助信息
if ($Help) {
    Get-Help $MyInvocation.MyCommand.Path -Full
    exit 0
}

# 设置错误处理
$ErrorActionPreference = "Stop"

# 全局变量
$StartTime = Get-Date
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)
$TestsDir = Join-Path $ProjectRoot "tests"

# 设置默认目录
$DefaultSourceDir = Join-Path $ProjectRoot "reports"
$DefaultTargetDir = Join-Path $TestsDir "reports"
$DefaultBackupDir = Join-Path $TestsDir "backups" "reports_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"

$ActualSourceDir = if ($SourceDir) { $SourceDir } else { $DefaultSourceDir }
$ActualTargetDir = if ($TargetDir) { $TargetDir } else { $DefaultTargetDir }
$ActualBackupDir = if ($BackupDir) { $BackupDir } else { $DefaultBackupDir }

$LogFile = Join-Path $TestsDir "logs" "migrate_reports_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

# 创建必要的目录
$DirsToCreate = @($ActualTargetDir, (Split-Path -Parent $LogFile))
if (-not $SkipBackup) {
    $DirsToCreate += $ActualBackupDir
}

foreach ($Dir in $DirsToCreate) {
    if (-not (Test-Path $Dir)) {
        if (-not $DryRun) {
            New-Item -Path $Dir -ItemType Directory -Force | Out-Null
        }
    }
}

# 日志记录函数
function Write-Log {
    param(
        [string]$Message,
        [string]$Level = "INFO"
    )
    
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $LogEntry = "[$Timestamp] [$Level] $Message"
    
    # 输出到控制台
    switch ($Level) {
        "SUCCESS" { Write-Host $LogEntry -ForegroundColor Green }
        "WARN" { Write-Host $LogEntry -ForegroundColor Yellow }
        "ERROR" { Write-Host $LogEntry -ForegroundColor Red }
        "DRY_RUN" { Write-Host $LogEntry -ForegroundColor Cyan }
        default { Write-Host $LogEntry }
    }
    
    # 写入日志文件
    if (-not $DryRun) {
        Add-Content -Path $LogFile -Value $LogEntry -Encoding UTF8
    }
}

# 错误处理函数
function Handle-Error {
    param([string]$ErrorMessage)
    
    Write-Log $ErrorMessage "ERROR"
    Write-Log "报告迁移失败，请检查错误信息" "ERROR"
    exit 1
}

# 确认操作函数
function Confirm-Action {
    param(
        [string]$Message,
        [string]$Title = "确认操作"
    )
    
    if ($Force -or $DryRun) {
        return $true
    }
    
    $Response = Read-Host "$Message (y/N)"
    return ($Response -eq 'y' -or $Response -eq 'Y' -or $Response -eq 'yes' -or $Response -eq 'Yes')
}

# 创建备份函数
function New-ReportBackup {
    if ($SkipBackup) {
        Write-Log "跳过备份创建" "WARN"
        return
    }
    
    Write-Log "创建报告备份..."
    
    if (-not (Test-Path $ActualSourceDir)) {
        Write-Log "源目录不存在，跳过备份: $ActualSourceDir" "WARN"
        return
    }
    
    try {
        if ($DryRun) {
            Write-Log "[DRY RUN] 将创建备份: $ActualSourceDir -> $ActualBackupDir" "DRY_RUN"
        } else {
            Copy-Item -Path $ActualSourceDir -Destination $ActualBackupDir -Recurse -Force
            Write-Log "备份创建成功: $ActualBackupDir" "SUCCESS"
        }
    }
    catch {
        Handle-Error "备份创建失败: $($_.Exception.Message)"
    }
}

# 迁移文件函数
function Move-ReportFiles {
    param(
        [string]$SourcePath,
        [string]$TargetPath,
        [string]$Description
    )
    
    if (-not (Test-Path $SourcePath)) {
        Write-Log "源路径不存在，跳过: $SourcePath" "WARN"
        return
    }
    
    Write-Log "迁移 $Description..."
    Write-Log "  源路径: $SourcePath"
    Write-Log "  目标路径: $TargetPath"
    
    try {
        if ($DryRun) {
            Write-Log "[DRY RUN] 将迁移: $SourcePath -> $TargetPath" "DRY_RUN"
        } else {
            # 确保目标目录存在
            $TargetParent = Split-Path -Parent $TargetPath
            if (-not (Test-Path $TargetParent)) {
                New-Item -Path $TargetParent -ItemType Directory -Force | Out-Null
            }
            
            # 如果目标已存在且不强制覆盖，询问用户
            if ((Test-Path $TargetPath) -and -not $Force) {
                if (-not (Confirm-Action "目标路径已存在，是否覆盖？ $TargetPath")) {
                    Write-Log "跳过迁移: $Description" "WARN"
                    return
                }
            }
            
            # 移动文件或目录
            if (Test-Path $TargetPath) {
                Remove-Item -Path $TargetPath -Recurse -Force
            }
            Move-Item -Path $SourcePath -Destination $TargetPath -Force
            Write-Log "$Description 迁移成功" "SUCCESS"
        }
    }
    catch {
        Write-Log "$Description 迁移失败: $($_.Exception.Message)" "ERROR"
    }
}

# 迁移覆盖率报告
function Move-CoverageReports {
    Write-Log "开始迁移覆盖率报告..."
    
    # 迁移HTML覆盖率报告
    $SourceHtmlCov = Join-Path $ActualSourceDir "coverage" "html"
    $TargetHtmlCov = Join-Path $ActualTargetDir "coverage" "html"
    Move-ReportFiles $SourceHtmlCov $TargetHtmlCov "HTML覆盖率报告"
    
    # 迁移XML覆盖率报告
    $SourceXmlCov = Join-Path $ActualSourceDir "coverage" "coverage.xml"
    $TargetXmlCov = Join-Path $ActualTargetDir "coverage" "coverage.xml"
    Move-ReportFiles $SourceXmlCov $TargetXmlCov "XML覆盖率报告"
    
    # 迁移JSON覆盖率报告
    $SourceJsonCov = Join-Path $ActualSourceDir "coverage" "coverage.json"
    $TargetJsonCov = Join-Path $ActualTargetDir "coverage" "coverage.json"
    Move-ReportFiles $SourceJsonCov $TargetJsonCov "JSON覆盖率报告"
    
    # 迁移整个覆盖率目录（如果存在其他文件）
    $SourceCovDir = Join-Path $ActualSourceDir "coverage"
    $TargetCovDir = Join-Path $ActualTargetDir "coverage"
    if ((Test-Path $SourceCovDir) -and (Get-ChildItem $SourceCovDir -ErrorAction SilentlyContinue)) {
        Move-ReportFiles $SourceCovDir $TargetCovDir "剩余覆盖率文件"
    }
}

# 迁移HTML测试报告
function Move-HtmlReports {
    Write-Log "开始迁移HTML测试报告..."
    
    $SourceHtmlReport = Join-Path $ActualSourceDir "html" "report.html"
    $TargetHtmlReport = Join-Path $ActualTargetDir "html" "report.html"
    Move-ReportFiles $SourceHtmlReport $TargetHtmlReport "HTML测试报告"
    
    # 迁移整个HTML目录
    $SourceHtmlDir = Join-Path $ActualSourceDir "html"
    $TargetHtmlDir = Join-Path $ActualTargetDir "html"
    if ((Test-Path $SourceHtmlDir) -and (Get-ChildItem $SourceHtmlDir -ErrorAction SilentlyContinue)) {
        Move-ReportFiles $SourceHtmlDir $TargetHtmlDir "HTML报告目录"
    }
}

# 迁移Allure报告
function Move-AllureReports {
    Write-Log "开始迁移Allure报告..."
    
    $SourceAllureDir = Join-Path $ActualSourceDir "allure"
    $TargetAllureDir = Join-Path $ActualTargetDir "allure"
    Move-ReportFiles $SourceAllureDir $TargetAllureDir "Allure报告"
}

# 迁移性能测试报告
function Move-PerformanceReports {
    Write-Log "开始迁移性能测试报告..."
    
    # 查找可能的性能报告目录
    $PossiblePerfDirs = @(
        "performance",
        "perf",
        "load_test",
        "stress_test"
    )
    
    foreach ($PerfDirName in $PossiblePerfDirs) {
        $SourcePerfDir = Join-Path $ActualSourceDir $PerfDirName
        $TargetPerfDir = Join-Path $ActualTargetDir "performance" $PerfDirName
        
        if (Test-Path $SourcePerfDir) {
            Move-ReportFiles $SourcePerfDir $TargetPerfDir "性能测试报告 ($PerfDirName)"
        }
    }
    
    # 迁移性能测试JSON文件
    $PerfJsonFiles = Get-ChildItem -Path $ActualSourceDir -Filter "*performance*.json" -ErrorAction SilentlyContinue
    foreach ($JsonFile in $PerfJsonFiles) {
        $TargetJsonPath = Join-Path $ActualTargetDir "performance" "metrics" $JsonFile.Name
        Move-ReportFiles $JsonFile.FullName $TargetJsonPath "性能测试JSON文件 ($($JsonFile.Name))"
    }
}

# 迁移安全测试报告
function Move-SecurityReports {
    Write-Log "开始迁移安全测试报告..."
    
    $SourceSecDir = Join-Path $ActualSourceDir "security"
    $TargetSecDir = Join-Path $ActualTargetDir "security"
    Move-ReportFiles $SourceSecDir $TargetSecDir "安全测试报告"
}

# 迁移其他报告文件
function Move-OtherReports {
    Write-Log "开始迁移其他报告文件..."
    
    if (-not (Test-Path $ActualSourceDir)) {
        Write-Log "源目录不存在: $ActualSourceDir" "WARN"
        return
    }
    
    # 获取所有剩余的文件和目录
    $RemainingItems = Get-ChildItem -Path $ActualSourceDir -ErrorAction SilentlyContinue
    
    foreach ($Item in $RemainingItems) {
        $TargetPath = Join-Path $ActualTargetDir "misc" $Item.Name
        Move-ReportFiles $Item.FullName $TargetPath "其他报告文件 ($($Item.Name))"
    }
}

# 清理空目录
function Remove-EmptyDirectories {
    Write-Log "清理空目录..."
    
    if (-not (Test-Path $ActualSourceDir)) {
        return
    }
    
    try {
        if ($DryRun) {
            Write-Log "[DRY RUN] 将清理空目录: $ActualSourceDir" "DRY_RUN"
        } else {
            # 递归删除空目录
            Get-ChildItem -Path $ActualSourceDir -Recurse -Directory | 
                Sort-Object FullName -Descending | 
                ForEach-Object {
                    if (-not (Get-ChildItem -Path $_.FullName -ErrorAction SilentlyContinue)) {
                        Remove-Item -Path $_.FullName -Force
                        Write-Log "删除空目录: $($_.FullName)" "SUCCESS"
                    }
                }
            
            # 如果源目录为空，删除它
            if (-not (Get-ChildItem -Path $ActualSourceDir -ErrorAction SilentlyContinue)) {
                Remove-Item -Path $ActualSourceDir -Force
                Write-Log "删除空的源目录: $ActualSourceDir" "SUCCESS"
            }
        }
    }
    catch {
        Write-Log "清理空目录时出错: $($_.Exception.Message)" "WARN"
    }
}

# 生成迁移报告
function New-MigrationReport {
    Write-Log "生成迁移报告..."
    
    $ReportPath = Join-Path $ActualTargetDir "migration_report_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"
    
    $ReportContent = @"
HarborAI测试报告迁移报告
========================

迁移时间: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
脚本版本: 1.0
操作模式: $(if ($DryRun) { "预览模式" } else { "实际迁移" })

目录配置:
- 源目录: $ActualSourceDir
- 目标目录: $ActualTargetDir
- 备份目录: $ActualBackupDir

迁移选项:
- 强制覆盖: $Force
- 跳过备份: $SkipBackup
- 详细输出: $Verbose

迁移结果:
- 覆盖率报告: $(if (Test-Path (Join-Path $ActualTargetDir "coverage")) { "✓ 已迁移" } else { "✗ 未找到" })
- HTML报告: $(if (Test-Path (Join-Path $ActualTargetDir "html")) { "✓ 已迁移" } else { "✗ 未找到" })
- Allure报告: $(if (Test-Path (Join-Path $ActualTargetDir "allure")) { "✓ 已迁移" } else { "✗ 未找到" })
- 性能报告: $(if (Test-Path (Join-Path $ActualTargetDir "performance")) { "✓ 已迁移" } else { "✗ 未找到" })
- 安全报告: $(if (Test-Path (Join-Path $ActualTargetDir "security")) { "✓ 已迁移" } else { "✗ 未找到" })

回滚说明:
如需回滚，请执行以下命令：
1. 删除目标目录: Remove-Item -Path "$ActualTargetDir" -Recurse -Force
2. 恢复备份: Copy-Item -Path "$ActualBackupDir" -Destination "$ActualSourceDir" -Recurse -Force

日志文件: $LogFile
"@

    if ($DryRun) {
        Write-Log "[DRY RUN] 将生成迁移报告: $ReportPath" "DRY_RUN"
        Write-Log $ReportContent
    } else {
        Set-Content -Path $ReportPath -Value $ReportContent -Encoding UTF8
        Write-Log "迁移报告已生成: $ReportPath" "SUCCESS"
    }
}

# 主执行函数
function Start-ReportMigration {
    Write-Log "开始HarborAI测试报告迁移..."
    Write-Log "源目录: $ActualSourceDir"
    Write-Log "目标目录: $ActualTargetDir"
    Write-Log "备份目录: $ActualBackupDir"
    
    if ($DryRun) {
        Write-Log "运行在预览模式，不会实际移动文件" "DRY_RUN"
    }
    
    # 检查源目录是否存在
    if (-not (Test-Path $ActualSourceDir)) {
        Write-Log "源目录不存在: $ActualSourceDir" "WARN"
        Write-Log "可能报告已经在正确位置，或者没有需要迁移的报告" "WARN"
        return
    }
    
    # 确认操作
    if (-not $DryRun -and -not (Confirm-Action "确认开始迁移报告？这将移动文件到新的目录结构")) {
        Write-Log "用户取消操作" "WARN"
        return
    }
    
    try {
        # 创建备份
        New-ReportBackup
        
        # 执行迁移
        Move-CoverageReports
        Move-HtmlReports
        Move-AllureReports
        Move-PerformanceReports
        Move-SecurityReports
        Move-OtherReports
        
        # 清理空目录
        Remove-EmptyDirectories
        
        # 生成迁移报告
        New-MigrationReport
        
        $EndTime = Get-Date
        $Duration = $EndTime - $StartTime
        
        if ($DryRun) {
            Write-Log "预览完成，耗时: $($Duration.TotalSeconds) 秒" "SUCCESS"
            Write-Log "使用 -DryRun:$false 参数执行实际迁移" "INFO"
        } else {
            Write-Log "报告迁移完成，耗时: $($Duration.TotalSeconds) 秒" "SUCCESS"
            Write-Log "备份位置: $ActualBackupDir" "INFO"
            Write-Log "日志文件: $LogFile" "INFO"
        }
    }
    catch {
        Handle-Error "迁移过程中发生错误: $($_.Exception.Message)"
    }
}

# 执行迁移
Start-ReportMigration