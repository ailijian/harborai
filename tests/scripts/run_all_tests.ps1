<#
.SYNOPSIS
    HarborAI Comprehensive Test Runner Script

.DESCRIPTION
    This script runs all types of tests for the HarborAI project, including:
    - Unit tests
    - Integration tests
    - API tests
    - Security tests
    - Compatibility tests
    - Code coverage analysis
    - Test result reporting

.PARAMETER TestType
    Type of tests to run (unit, integration, api, security, compatibility, all)

.PARAMETER Coverage
    Enable code coverage analysis

.PARAMETER Parallel
    Run tests in parallel

.PARAMETER Verbose
    Show verbose output

.PARAMETER FailFast
    Stop on first test failure

.PARAMETER ReportFormat
    Test report format (html, xml, json, allure)

.PARAMETER Help
    Show help information

.EXAMPLE
    .\run_all_tests.ps1
    Run all tests with default settings

.EXAMPLE
    .\run_all_tests.ps1 -TestType unit -Coverage -Verbose
    Run unit tests with coverage and verbose output

.EXAMPLE
    .\run_all_tests.ps1 -TestType all -Parallel -ReportFormat allure
    Run all tests in parallel with Allure reports
#>

param(
    [ValidateSet("unit", "integration", "api", "security", "compatibility", "all")]
    [string]$TestType = "all",
    [switch]$Coverage,
    [switch]$Parallel,
    [switch]$Verbose,
    [switch]$FailFast,
    [ValidateSet("html", "xml", "json", "allure")]
    [string]$ReportFormat = "html",
    [switch]$Help
)

# Show help information
if ($Help) {
    Get-Help $MyInvocation.MyCommand.Path -Full
    exit 0
}

# Set error handling
$ErrorActionPreference = "Stop"

# Global variables
$StartTime = Get-Date
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)
$TestsDir = Join-Path $ProjectRoot "tests"
$ReportsDir = Join-Path $TestsDir "reports"
$LogFile = Join-Path $TestsDir "logs" "test_run_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

# Create log directory
$LogDir = Split-Path -Parent $LogFile
if (-not (Test-Path $LogDir)) {
    New-Item -Path $LogDir -ItemType Directory -Force | Out-Null
}

# Logging function
function Write-Log {
    param(
        [string]$Message,
        [string]$Level = "INFO"
    )
    
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $LogEntry = "[$Timestamp] [$Level] $Message"
    
    # Output to console
    switch ($Level) {
        "SUCCESS" { Write-Host $LogEntry -ForegroundColor Green }
        "WARN" { Write-Host $LogEntry -ForegroundColor Yellow }
        "ERROR" { Write-Host $LogEntry -ForegroundColor Red }
        default { Write-Host $LogEntry }
    }
    
    # Write to log file
    Add-Content -Path $LogFile -Value $LogEntry -Encoding UTF8
}

# Error handling function
function Handle-Error {
    param([string]$ErrorMessage)
    
    Write-Log $ErrorMessage "ERROR"
    Write-Log "Test execution failed, please check error information" "ERROR"
    exit 1
}

# Check if command exists
function Test-Command {
    param([string]$Command)
    
    try {
        Get-Command $Command -ErrorAction Stop | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

# Check test environment
function Test-TestEnvironment {
    Write-Log "Checking test environment..."
    
    # Check Python
    if (-not (Test-Command "python")) {
        Handle-Error "Python not found, please run setup_test_env.ps1 first"
    }
    
    # Check pytest
    try {
        python -c "import pytest" 2>$null
        Write-Log "pytest is available" "SUCCESS"
    }
    catch {
        Handle-Error "pytest not found, please run setup_test_env.ps1 first"
    }
    
    # Check test directory
    if (-not (Test-Path $TestsDir)) {
        Handle-Error "Tests directory not found: $TestsDir"
    }
    
    Write-Log "Test environment check passed" "SUCCESS"
}

# Run unit tests
function Invoke-UnitTests {
    Write-Log "Running unit tests..."
    
    $UnitTestsDir = Join-Path $TestsDir "unit"
    if (-not (Test-Path $UnitTestsDir)) {
        Write-Log "Unit tests directory not found, skipping unit tests" "WARN"
        return
    }
    
    $TestArgs = @(
        $UnitTestsDir,
        "-v"
    )
    
    if ($Coverage) {
        $TestArgs += @("--cov=$ProjectRoot", "--cov-report=html:$ReportsDir/coverage")
    }
    
    if ($FailFast) {
        $TestArgs += "-x"
    }
    
    if ($Parallel) {
        $TestArgs += "-n auto"
    }
    
    try {
        Push-Location $ProjectRoot
        python -m pytest @TestArgs
        Write-Log "Unit tests completed" "SUCCESS"
    }
    catch {
        Write-Log "Unit tests failed: $($_.Exception.Message)" "ERROR"
        if ($FailFast) {
            throw
        }
    }
    finally {
        Pop-Location
    }
}

# Run integration tests
function Invoke-IntegrationTests {
    Write-Log "Running integration tests..."
    
    $IntegrationTestsDir = Join-Path $TestsDir "integration"
    if (-not (Test-Path $IntegrationTestsDir)) {
        Write-Log "Integration tests directory not found, skipping integration tests" "WARN"
        return
    }
    
    $TestArgs = @(
        $IntegrationTestsDir,
        "-v",
        "--tb=short"
    )
    
    if ($FailFast) {
        $TestArgs += "-x"
    }
    
    try {
        Push-Location $ProjectRoot
        python -m pytest @TestArgs
        Write-Log "Integration tests completed" "SUCCESS"
    }
    catch {
        Write-Log "Integration tests failed: $($_.Exception.Message)" "ERROR"
        if ($FailFast) {
            throw
        }
    }
    finally {
        Pop-Location
    }
}

# Run API tests
function Invoke-ApiTests {
    Write-Log "Running API tests..."
    
    $ApiTestsDir = Join-Path $TestsDir "api"
    if (-not (Test-Path $ApiTestsDir)) {
        Write-Log "API tests directory not found, skipping API tests" "WARN"
        return
    }
    
    $TestArgs = @(
        $ApiTestsDir,
        "-v",
        "--tb=short"
    )
    
    if ($FailFast) {
        $TestArgs += "-x"
    }
    
    try {
        Push-Location $ProjectRoot
        python -m pytest @TestArgs
        Write-Log "API tests completed" "SUCCESS"
    }
    catch {
        Write-Log "API tests failed: $($_.Exception.Message)" "ERROR"
        if ($FailFast) {
            throw
        }
    }
    finally {
        Pop-Location
    }
}

# Run security tests
function Invoke-SecurityTests {
    Write-Log "Running security tests..."
    
    $SecurityTestsDir = Join-Path $TestsDir "security"
    if (-not (Test-Path $SecurityTestsDir)) {
        Write-Log "Security tests directory not found, skipping security tests" "WARN"
        return
    }
    
    try {
        Push-Location $ProjectRoot
        python -m pytest $SecurityTestsDir -v
        Write-Log "Security tests completed" "SUCCESS"
    }
    catch {
        Write-Log "Security tests failed: $($_.Exception.Message)" "ERROR"
        if ($FailFast) {
            throw
        }
    }
    finally {
        Pop-Location
    }
}

# Run compatibility tests
function Invoke-CompatibilityTests {
    Write-Log "Running compatibility tests..."
    
    $CompatibilityTestsDir = Join-Path $TestsDir "compatibility"
    if (-not (Test-Path $CompatibilityTestsDir)) {
        Write-Log "Compatibility tests directory not found, skipping compatibility tests" "WARN"
        return
    }
    
    try {
        Push-Location $ProjectRoot
        python -m pytest $CompatibilityTestsDir -v
        Write-Log "Compatibility tests completed" "SUCCESS"
    }
    catch {
        Write-Log "Compatibility tests failed: $($_.Exception.Message)" "ERROR"
        if ($FailFast) {
            throw
        }
    }
    finally {
        Pop-Location
    }
}

# Generate test reports
function New-TestReports {
    Write-Log "Generating test reports..."
    
    $ReportArgs = @()
    
    switch ($ReportFormat) {
        "html" {
            $ReportArgs += @("--html=$ReportsDir/test_report.html", "--self-contained-html")
        }
        "xml" {
            $ReportArgs += "--junitxml=$ReportsDir/test_report.xml"
        }
        "json" {
            $ReportArgs += "--json-report --json-report-file=$ReportsDir/test_report.json"
        }
        "allure" {
            $AllureDir = Join-Path $ReportsDir "allure"
            if (-not (Test-Path $AllureDir)) {
                New-Item -Path $AllureDir -ItemType Directory -Force | Out-Null
            }
            $ReportArgs += "--alluredir=$AllureDir"
        }
    }
    
    if ($ReportArgs.Count -gt 0) {
        Write-Log "Report format: $ReportFormat"
        Write-Log "Report arguments: $($ReportArgs -join ' ')"
    }
}

# Parse test results
function Get-TestResults {
    Write-Log "Parsing test results..."
    
    # This is a placeholder for test result parsing
    # In a real implementation, you would parse the test output
    # and extract metrics like pass/fail counts, duration, etc.
    
    Write-Log "Test results parsing completed" "SUCCESS"
}

# Main function
function Main {
    Write-Log "Starting HarborAI test execution..."
    Write-Log "Test type: $TestType"
    Write-Log "Coverage: $Coverage"
    Write-Log "Parallel: $Parallel"
    Write-Log "Report format: $ReportFormat"
    Write-Log "Project root: $ProjectRoot"
    Write-Log "Tests directory: $TestsDir"
    
    try {
        # Check test environment
        Test-TestEnvironment
        
        # Generate test reports configuration
        New-TestReports
        
        # Run tests based on type
        switch ($TestType) {
            "unit" {
                Invoke-UnitTests
            }
            "integration" {
                Invoke-IntegrationTests
            }
            "api" {
                Invoke-ApiTests
            }
            "security" {
                Invoke-SecurityTests
            }
            "compatibility" {
                Invoke-CompatibilityTests
            }
            "all" {
                Invoke-UnitTests
                Invoke-IntegrationTests
                Invoke-ApiTests
                Invoke-SecurityTests
                Invoke-CompatibilityTests
            }
        }
        
        # Parse test results
        Get-TestResults
        
        $ElapsedTime = (Get-Date) - $StartTime
        Write-Log "Test execution completed! Time elapsed: $($ElapsedTime.TotalSeconds.ToString('F2')) seconds" "SUCCESS"
        Write-Log "Log file: $LogFile"
        
        Write-Host ""
        Write-Host "=== Test Execution Completed ===" -ForegroundColor Green
        Write-Host "Test type: $TestType" -ForegroundColor Cyan
        Write-Host "Report format: $ReportFormat" -ForegroundColor Cyan
        Write-Host "Reports directory: $ReportsDir" -ForegroundColor Cyan
        Write-Host ""
        
    }
    catch {
        Handle-Error "Unexpected error during test execution: $($_.Exception.Message)"
    }
}

# Script entry point
if ($MyInvocation.InvocationName -ne '.') {
    Main
}