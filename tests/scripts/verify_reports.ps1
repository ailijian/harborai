# HarborAI Report Verification Script
# Verifies unified report path implementation

param(
    [switch]$Verbose,
    [switch]$FixIssues,
    [switch]$GenerateReport = $true,
    [switch]$Help
)

if ($Help) {
    Write-Host "HarborAI Report Verification Script" -ForegroundColor Green
    Write-Host ""
    Write-Host "Usage:"
    Write-Host "    .\verify_reports.ps1 [parameters]"
    Write-Host ""
    Write-Host "Parameters:"
    Write-Host "    -Verbose        Show detailed output"
    Write-Host "    -FixIssues      Auto-fix discovered issues"
    Write-Host "    -GenerateReport Generate verification report (default: true)"
    Write-Host "    -Help           Show this help"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "    .\verify_reports.ps1 -Verbose"
    Write-Host "    .\verify_reports.ps1 -FixIssues -Verbose"
    exit 0
}

# Global variables
$StartTime = Get-Date
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)
$TestsDir = Join-Path $ProjectRoot "tests"
$ActualReportsDir = Join-Path $TestsDir "reports"
$LogFile = Join-Path $ActualReportsDir "verification_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

# Verification results storage
$VerificationResults = @{
    DirectoryStructure = @{
        RootExists = $false
        RequiredDirs = @()
        ExistingDirs = @()
    }
    PytestConfig = @{
        FileExists = $false
        ConfigCorrect = $false
    }
    PowerShellScripts = @{
        Scripts = @{}
    }
    UnifiedManager = @{
        FileExists = $false
        FunctionalityWorks = $false
    }
    PerformanceIntegration = @{
        Files = @{}
    }
    TestExecution = $false
    Issues = @()
    Successes = @()
    OverallStatus = "UNKNOWN"
}

# Logging function
function Write-Log {
    param(
        [string]$Message,
        [string]$Level = "INFO"
    )
    
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $LogMessage = "[$Timestamp] [$Level] $Message"
    
    # Ensure log directory exists
    $LogDir = Split-Path -Parent $LogFile
    if (-not (Test-Path $LogDir)) {
        New-Item -Path $LogDir -ItemType Directory -Force | Out-Null
    }
    
    Add-Content -Path $LogFile -Value $LogMessage -Encoding UTF8
    
    if ($Verbose) {
        switch ($Level) {
            "ERROR" { Write-Host $LogMessage -ForegroundColor Red }
            "WARN" { Write-Host $LogMessage -ForegroundColor Yellow }
            "SUCCESS" { Write-Host $LogMessage -ForegroundColor Green }
            default { Write-Host $LogMessage }
        }
    }
}

# Add issue
function Add-Issue {
    param(
        [string]$Category,
        [string]$Description,
        [string]$Severity = "MEDIUM",
        [string]$Recommendation = ""
    )
    
    $Issue = @{
        Category = $Category
        Description = $Description
        Severity = $Severity
        Recommendation = $Recommendation
        Timestamp = Get-Date
    }
    
    $VerificationResults.Issues += $Issue
    Write-Log "Issue: [$Category] $Description (Severity: $Severity)" "WARN"
}

# Add success record
function Add-Success {
    param(
        [string]$Category,
        [string]$Description
    )
    
    $Success = @{
        Category = $Category
        Description = $Description
        Timestamp = Get-Date
    }
    
    $VerificationResults.Successes += $Success
    Write-Log "Success: [$Category] $Description" "SUCCESS"
}

# Verify directory structure
function Test-DirectoryStructure {
    Write-Log "Verifying directory structure..."
    
    $RequiredDirs = @(
        "tests",
        "tests/reports",
        "tests/reports/coverage",
        "tests/reports/coverage/html",
        "tests/reports/html",
        "tests/reports/allure",
        "tests/reports/performance",
        "tests/reports/security",
        "tests/utils",
        "tests/performance"
    )
    
    $VerificationResults.DirectoryStructure.RequiredDirs = $RequiredDirs
    $VerificationResults.DirectoryStructure.RootExists = Test-Path $TestsDir
    
    if (-not $VerificationResults.DirectoryStructure.RootExists) {
        Add-Issue "Directory Structure" "Test root directory does not exist: $TestsDir" "HIGH" "Create test directory structure"
        return
    }
    
    foreach ($Dir in $RequiredDirs) {
        $FullPath = Join-Path $ProjectRoot $Dir
        if (Test-Path $FullPath) {
            $VerificationResults.DirectoryStructure.ExistingDirs += $Dir
            Add-Success "Directory Structure" "Directory exists: $Dir"
        } else {
            Add-Issue "Directory Structure" "Directory missing: $Dir" "MEDIUM" "Create missing directory"
            
            if ($FixIssues) {
                try {
                    New-Item -Path $FullPath -ItemType Directory -Force | Out-Null
                    Write-Log "Created directory: $Dir" "SUCCESS"
                    $VerificationResults.DirectoryStructure.ExistingDirs += $Dir
                } catch {
                    Write-Log "Cannot create directory $Dir : $($_.Exception.Message)" "ERROR"
                }
            }
        }
    }
}

# Verify pytest configuration
function Test-PytestConfig {
    Write-Log "Verifying pytest configuration..."
    
    $PytestConfigPath = Join-Path $ProjectRoot "pytest.ini"
    
    if (-not (Test-Path $PytestConfigPath)) {
        Add-Issue "Pytest Config" "pytest.ini file does not exist" "HIGH" "Create pytest.ini configuration file"
        $VerificationResults.PytestConfig.FileExists = $false
        return
    }
    
    $VerificationResults.PytestConfig.FileExists = $true
    Add-Success "Pytest Config" "pytest.ini file exists"
    
    $ConfigContent = Get-Content -Path $PytestConfigPath -Raw
    
    # Check key configuration items
    $RequiredConfigs = @(
        "--cov-report=html:tests/reports/coverage/html",
        "--cov-report=xml:tests/reports/coverage/coverage.xml",
        "--html=tests/reports/html/report.html",
        "--alluredir=tests/reports/allure"
    )
    
    $ConfigValid = $true
    foreach ($Config in $RequiredConfigs) {
        if ($ConfigContent -notmatch [regex]::Escape($Config)) {
            $ConfigValid = $false
            Add-Issue "Pytest Config" "Missing configuration item: $Config" "MEDIUM" "Add missing configuration item to pytest.ini"
        }
    }
    
    $VerificationResults.PytestConfig.ConfigCorrect = $ConfigValid
    
    if ($ConfigValid) {
        Add-Success "Pytest Config" "All required configuration items exist"
    }
}

# Verify PowerShell scripts
function Test-PowerShellScripts {
    Write-Log "Verifying PowerShell scripts..."
    
    $ScriptsDir = Join-Path $TestsDir "scripts"
    $ScriptFiles = @(
        "run_all_tests.ps1",
        "generate_reports.ps1",
        "run_performance_tests.ps1",
        "setup_test_env.ps1",
        "cleanup_test_env.ps1"
    )
    
    $VerificationResults.PowerShellScripts.Scripts = @{}
    
    foreach ($ScriptFile in $ScriptFiles) {
        $ScriptPath = Join-Path $ScriptsDir $ScriptFile
        
        if (-not (Test-Path $ScriptPath)) {
            $VerificationResults.PowerShellScripts.Scripts[$ScriptFile] = $false
            Add-Issue "PowerShell Scripts" "Script file does not exist: $ScriptFile" "MEDIUM" "Ensure all required PowerShell scripts exist"
            continue
        }
        
        $ScriptContent = Get-Content -Path $ScriptPath -Raw
        
        # Check basic path variables
        $RequiredVariables = @(
            '$TestsDir = Join-Path $ProjectRoot "tests"',
            '$ProjectRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)'
        )
        
        $ScriptValid = $true
        foreach ($Variable in $RequiredVariables) {
            if ($ScriptContent -notmatch [regex]::Escape($Variable)) {
                $ScriptValid = $false
                Add-Issue "PowerShell Scripts" "$ScriptFile missing required variable definition: $Variable" "MEDIUM" "Add missing variable definition"
            }
        }
        
        # Check report directory configuration
        if ($ScriptFile -eq "run_all_tests.ps1" -or $ScriptFile -eq "generate_reports.ps1") {
            # Check for unified report path patterns
            $UnifiedPatterns = @(
                'Join-Path.*"tests".*"reports"',
                '\$TestsDir.*"reports"',
                '\$DefaultOutputDir.*Join-Path.*\$TestsDir.*"reports"'
            )
            
            $HasUnifiedPattern = $false
            foreach ($Pattern in $UnifiedPatterns) {
                if ($ScriptContent -match $Pattern) {
                    $HasUnifiedPattern = $true
                    break
                }
            }
            
            if (-not $HasUnifiedPattern) {
                $ScriptValid = $false
                Add-Issue "PowerShell Scripts" "$ScriptFile report directory configuration incorrect" "HIGH" "Update report directory configuration to unified path"
            }
        }
        
        $VerificationResults.PowerShellScripts.Scripts[$ScriptFile] = $ScriptValid
        
        if ($ScriptValid) {
            Add-Success "PowerShell Scripts" "$ScriptFile configuration correct"
        }
    }
}

# Verify unified report manager
function Test-UnifiedManager {
    Write-Log "Verifying unified report manager..."
    
    $UtilsDir = Join-Path $TestsDir "utils"
    $ManagerPath = Join-Path $UtilsDir "unified_report_manager.py"
    
    if (-not (Test-Path $ManagerPath)) {
        Add-Issue "Unified Manager" "unified_report_manager.py file does not exist" "HIGH" "Create unified report manager file"
        $VerificationResults.UnifiedManager.FileExists = $false
        return
    }
    
    $VerificationResults.UnifiedManager.FileExists = $true
    Add-Success "Unified Manager" "unified_report_manager.py file exists"
    
    # Try to import and test manager
    try {
        Push-Location $ProjectRoot
        
        $TestScript = @"
import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'tests'))

try:
    from utils.unified_report_manager import get_report_manager, get_coverage_report_path
    
    # Test basic functionality
    manager = get_report_manager()
    coverage_path = get_coverage_report_path('html')
    
    print(f"Manager initialized: {manager is not None}")
    print(f"Coverage path: {coverage_path}")
    print("SUCCESS")
except Exception as e:
    print(f"ERROR: {e}")
"@
        
        $TempScript = Join-Path $env:TEMP "test_manager.py"
        Set-Content -Path $TempScript -Value $TestScript -Encoding UTF8
        
        $Result = python $TempScript 2>&1
        
        if ($Result -match "SUCCESS") {
            Add-Success "Unified Manager" "Manager functionality works"
            $VerificationResults.UnifiedManager.FunctionalityWorks = $true
        } else {
            Add-Issue "Unified Manager" "Manager functionality test failed: $Result" "HIGH" "Check manager code and dependencies"
            $VerificationResults.UnifiedManager.FunctionalityWorks = $false
        }
        
        Remove-Item -Path $TempScript -Force -ErrorAction SilentlyContinue
    }
    catch {
        Add-Issue "Unified Manager" "Cannot test manager functionality: $($_.Exception.Message)" "HIGH" "Check Python environment and code"
        $VerificationResults.UnifiedManager.FunctionalityWorks = $false
    }
    finally {
        Pop-Location
    }
}

# Verify performance test integration
function Test-PerformanceIntegration {
    Write-Log "Verifying performance test integration..."
    
    $PerformanceDir = Join-Path $TestsDir "performance"
    $PerformanceFiles = @{
        "performance_report_generator.py" = Join-Path $PerformanceDir "performance_report_generator.py"
        "simple_performance_test.py" = Join-Path $PerformanceDir "simple_performance_test.py"
    }
    
    $VerificationResults.PerformanceIntegration.Files = @{}
    
    foreach ($FileName in $PerformanceFiles.Keys) {
        $FilePath = $PerformanceFiles[$FileName]
        
        if (-not (Test-Path $FilePath)) {
            $VerificationResults.PerformanceIntegration.Files[$FileName] = $false
            Add-Issue "Performance Integration" "File does not exist: $FileName" "MEDIUM" "Ensure performance test files exist"
            continue
        }
        
        $FileContent = Get-Content -Path $FilePath -Raw
        
        # Check if unified report manager is used
        if ($FileContent -match "get_performance_report_path" -or $FileContent -match "unified_report_manager") {
            $VerificationResults.PerformanceIntegration.Files[$FileName] = $true
            Add-Success "Performance Integration" "$FileName integrated with unified report manager"
        } else {
            $VerificationResults.PerformanceIntegration.Files[$FileName] = $false
            Add-Issue "Performance Integration" "$FileName not integrated with unified report manager" "MEDIUM" "Update file to use unified report manager"
        }
    }
}

# Run test verification
function Test-TestExecution {
    Write-Log "Verifying test execution..."
    
    try {
        Push-Location $ProjectRoot
        
        # Run a simple test to verify configuration
        $TestResult = python -m pytest tests/test_unified_report_manager.py --tb=short -q 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            Add-Success "Test Execution" "Unified report manager test passed"
            $VerificationResults.TestExecution = $true
        } else {
            Add-Issue "Test Execution" "Unified report manager test failed: $TestResult" "HIGH" "Check test code and environment configuration"
            $VerificationResults.TestExecution = $false
        }
    }
    catch {
        Add-Issue "Test Execution" "Cannot run test: $($_.Exception.Message)" "HIGH" "Check Python environment and pytest configuration"
        $VerificationResults.TestExecution = $false
    }
    finally {
        Pop-Location
    }
}

# Generate verification report
function New-VerificationReport {
    if (-not $GenerateReport) {
        return
    }
    
    Write-Log "Generating verification report..."
    
    $ReportPath = Join-Path $ActualReportsDir "verification_report_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"
    
    # Ensure report directory exists
    $ReportDir = Split-Path -Parent $ReportPath
    if (-not (Test-Path $ReportDir)) {
        New-Item -Path $ReportDir -ItemType Directory -Force | Out-Null
    }
    
    $IssueCount = $VerificationResults.Issues.Count
    $OverallStatus = if ($IssueCount -eq 0) { "PASS" } else { "FAIL" }
    $VerificationResults.OverallStatus = $OverallStatus
    
    # Build text report content
    $ReportContent = "HarborAI Test Report Verification Report`n"
    $ReportContent += "========================================`n`n"
    $ReportContent += "Generated: $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))`n"
    $ReportContent += "Scope: Unified report path implementation verification`n`n"
    $ReportContent += "Overall Status: $OverallStatus"

    if ($IssueCount -gt 0) {
        $ReportContent += " ($IssueCount issues need attention)"
    } else {
        $ReportContent += " (all checks passed)"
    }

    $ReportContent += "`n`nVerification Overview`n"
    $ReportContent += "====================`n"
    $ReportContent += "Directory Structure: $(if ($VerificationResults.DirectoryStructure.RootExists) { 'PASS' } else { 'FAIL' })`n"
    $ReportContent += "Pytest Config: $(if ($VerificationResults.PytestConfig.FileExists) { 'PASS' } else { 'FAIL' })`n"
    $ReportContent += "Unified Manager: $(if ($VerificationResults.UnifiedManager.FileExists) { 'PASS' } else { 'FAIL' })`n"
    $ReportContent += "Test Execution: $(if ($VerificationResults.TestExecution) { 'PASS' } else { 'FAIL' })`n"

    if ($VerificationResults.Issues.Count -gt 0) {
        $ReportContent += "`n`nIssues Found`n"
        $ReportContent += "============`n"
        foreach ($Issue in $VerificationResults.Issues) {
            $ReportContent += "Category: $($Issue.Category)`n"
            $ReportContent += "Description: $($Issue.Description)`n"
            $ReportContent += "Severity: $($Issue.Severity)`n"
            $ReportContent += "Recommendation: $($Issue.Recommendation)`n"
            $ReportContent += "---`n`n"
        }
    }

    $ReportContent += "`n`nSystem Information`n"
    $ReportContent += "==================`n"
    $ReportContent += "Project Root: $ProjectRoot`n"
    $ReportContent += "Tests Directory: $TestsDir`n"
    $ReportContent += "Reports Directory: $ActualReportsDir`n"
    $ReportContent += "Verification Time: $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))`n"
    $ReportContent += "Log File: $LogFile`n"

    Set-Content -Path $ReportPath -Value $ReportContent -Encoding UTF8
    Write-Log "Verification report generated: $ReportPath" "SUCCESS"
}

# Main execution function
function Start-ReportVerification {
    Write-Log "Starting HarborAI test report verification..."
    Write-Log "Reports directory: $ActualReportsDir"
    Write-Log "Project root: $ProjectRoot"
    
    if ($FixIssues) {
        Write-Log "Auto-fix mode enabled" "WARN"
    }
    
    try {
        # Execute verifications
        Test-DirectoryStructure
        Test-PytestConfig
        Test-PowerShellScripts
        Test-UnifiedManager
        Test-PerformanceIntegration
        Test-TestExecution
        
        # Generate verification report
        New-VerificationReport
        
        $EndTime = Get-Date
        $Duration = $EndTime - $StartTime
        
        $IssueCount = $VerificationResults.Issues.Count
        $OverallStatus = if ($IssueCount -eq 0) { "PASS" } else { "FAIL" }
        
        Write-Log "Verification completed in $($Duration.TotalSeconds) seconds" "SUCCESS"
        Write-Log "Overall status: $OverallStatus" $(if ($OverallStatus -eq "PASS") { "SUCCESS" } else { "ERROR" })
        
        if ($IssueCount -gt 0) {
            Write-Log "Found $IssueCount issues that need attention" "WARN"
            Write-Log "Use -FixIssues parameter to auto-fix issues" "INFO"
        } else {
            Write-Log "All verification checks passed!" "SUCCESS"
        }
        
        Write-Log "Log file: $LogFile" "INFO"
        
        # Return appropriate exit code
        if ($OverallStatus -eq "FAIL") {
            exit 1
        } else {
            exit 0
        }
    }
    catch {
        Write-Log "Error during verification: $($_.Exception.Message)" "ERROR"
        exit 1
    }
}

# Start verification
Start-ReportVerification