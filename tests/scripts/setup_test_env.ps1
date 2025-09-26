<#
.SYNOPSIS
    HarborAI Test Environment Setup Script

.DESCRIPTION
    This script sets up the test environment for HarborAI project, including:
    - Python environment check and dependency installation
    - Test environment variables configuration
    - Docker test environment startup (optional)
    - Test database initialization
    - Test directory structure creation

.PARAMETER SkipDependencies
    Skip Python dependency installation

.PARAMETER SkipDocker
    Skip Docker environment setup

.PARAMETER Verbose
    Show verbose output

.PARAMETER Help
    Show help information

.EXAMPLE
    .\setup_test_env.ps1
    Run test environment setup with default settings

.EXAMPLE
    .\setup_test_env.ps1 -SkipDocker -Verbose
    Skip Docker setup and show verbose output
#>

param(
    [switch]$SkipDependencies,
    [switch]$SkipDocker,
    [switch]$Verbose,
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
$LogFile = Join-Path $TestsDir "logs" "setup_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

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
    Write-Log "Test environment setup failed, please check error information" "ERROR"
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

# Check Python environment
function Test-PythonEnvironment {
    Write-Log "Checking Python environment..."
    
    if (-not (Test-Command "python")) {
        Handle-Error "Python not found, please install Python 3.8+"
    }
    
    $PythonVersion = python --version 2>&1
    Write-Log "Found Python version: $PythonVersion"
    
    Write-Log "Python environment check passed" "SUCCESS"
}

# Install Python dependencies
function Install-PythonDependencies {
    if ($SkipDependencies) {
        Write-Log "Skipping dependency installation"
        return
    }
    
    Write-Log "Installing Python dependencies..."
    
    # Switch to project root directory
    Push-Location $ProjectRoot
    
    try {
        # Upgrade pip
        Write-Log "Upgrading pip..."
        python -m pip install --upgrade pip
        
        # Install project dependencies
        if (Test-Path "requirements.txt") {
            Write-Log "Installing project dependencies (requirements.txt)..."
            python -m pip install -r requirements.txt
        }
        
        # Install test dependencies
        $TestRequirements = Join-Path $TestsDir "requirements-test.txt"
        if (Test-Path $TestRequirements) {
            Write-Log "Installing test dependencies (requirements-test.txt)..."
            python -m pip install -r $TestRequirements
        }
        
        Write-Log "Python dependencies installation completed" "SUCCESS"
    }
    catch {
        Handle-Error "Python dependencies installation failed: $($_.Exception.Message)"
    }
    finally {
        Pop-Location
    }
}

# Set environment variables
function Set-TestEnvironmentVariables {
    Write-Log "Setting test environment variables..."
    
    # Basic test environment variables
    $env:HARBORAI_TEST_MODE = "true"
    $env:HARBORAI_LOG_LEVEL = "DEBUG"
    $env:HARBORAI_ENABLE_LOGGING = "false"
    $env:PYTHONPATH = $ProjectRoot
    
    # Test database configuration (if using Docker)
    if (-not $SkipDocker) {
        $env:HARBORAI_TEST_DB_HOST = "localhost"
        $env:HARBORAI_TEST_DB_PORT = "5432"
        $env:HARBORAI_TEST_DB_NAME = "harborai_test"
        $env:HARBORAI_TEST_DB_USER = "test_user"
        $env:HARBORAI_TEST_DB_PASSWORD = "test_password"
    }
    
    # Create .env.test file
    $EnvTestFile = Join-Path $TestsDir ".env.test"
    $EnvContent = @"
# HarborAI Test Environment Configuration
# Auto-generated at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')

# Basic Configuration
HARBORAI_TEST_MODE=true
HARBORAI_LOG_LEVEL=DEBUG
HARBORAI_ENABLE_LOGGING=false

# Database Configuration (Docker)
HARBORAI_TEST_DB_HOST=localhost
HARBORAI_TEST_DB_PORT=5432
HARBORAI_TEST_DB_NAME=harborai_test
HARBORAI_TEST_DB_USER=test_user
HARBORAI_TEST_DB_PASSWORD=test_password

# Performance Test Configuration
PERFORMANCE_TEST_ITERATIONS=100
PERFORMANCE_TEST_CONCURRENT_USERS=10
PERFORMANCE_TEST_TIMEOUT=30
"@
    
    $EnvContent | Out-File -FilePath $EnvTestFile -Encoding UTF8
    Write-Log "Environment variables configuration completed, config file: $EnvTestFile" "SUCCESS"
}

# Create necessary directories
function New-TestDirectories {
    Write-Log "Creating test directory structure..."
    
    $Directories = @(
        $ReportsDir,
        (Join-Path $ReportsDir "html"),
        (Join-Path $ReportsDir "allure"),
        (Join-Path $ReportsDir "performance"),
        (Join-Path $ReportsDir "coverage"),
        (Join-Path $TestsDir "data"),
        (Join-Path $TestsDir "data" "schemas"),
        (Join-Path $TestsDir "data" "mock_responses"),
        (Join-Path $TestsDir "data" "test_cases"),
        (Join-Path $TestsDir "data" "performance_baselines")
    )
    
    foreach ($Dir in $Directories) {
        if (-not (Test-Path $Dir)) {
            New-Item -Path $Dir -ItemType Directory -Force | Out-Null
            if ($Verbose) {
                Write-Log "Created directory: $Dir"
            }
        }
    }
    
    Write-Log "Test directory structure creation completed" "SUCCESS"
}

# Check Docker environment
function Test-DockerEnvironment {
    if ($SkipDocker) {
        Write-Log "Skipping Docker environment check"
        return $false
    }
    
    Write-Log "Checking Docker environment..."
    
    if (-not (Test-Command "docker")) {
        Write-Log "Docker not found, skipping Docker setup" "WARN"
        return $false
    }
    
    if (-not (Test-Command "docker-compose")) {
        Write-Log "docker-compose not found, skipping Docker setup" "WARN"
        return $false
    }
    
    # Check if Docker is running
    try {
        docker info | Out-Null
        Write-Log "Docker environment check passed" "SUCCESS"
        return $true
    }
    catch {
        Write-Log "Docker is not running, please start Docker Desktop" "WARN"
        return $false
    }
}

# Start Docker test environment
function Start-DockerTestEnvironment {
    if ($SkipDocker -or -not (Test-DockerEnvironment)) {
        return
    }
    
    Write-Log "Starting Docker test environment..."
    
    $DockerComposeFile = Join-Path $TestsDir "docker-compose.test.yml"
    
    # If docker-compose file doesn't exist, create a basic one
    if (-not (Test-Path $DockerComposeFile)) {
        Write-Log "Creating Docker Compose configuration file..."
        $DockerContent = @"
version: '3.8'

services:
  postgres-test:
    image: postgres:13
    container_name: harborai-test-db
    environment:
      POSTGRES_DB: harborai_test
      POSTGRES_USER: test_user
      POSTGRES_PASSWORD: test_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_test_data:/var/lib/postgresql/data

volumes:
  postgres_test_data:
"@
        $DockerContent | Out-File -FilePath $DockerComposeFile -Encoding UTF8
    }
    
    # Start Docker containers
    Push-Location $TestsDir
    try {
        Write-Log "Starting PostgreSQL test database..."
        docker-compose -f docker-compose.test.yml up -d
        
        # Wait for database to start
        Write-Log "Waiting for database to start..."
        Start-Sleep -Seconds 10
        
        Write-Log "Docker test environment startup completed" "SUCCESS"
    }
    catch {
        Write-Log "Docker test environment startup failed: $($_.Exception.Message)" "WARN"
    }
    finally {
        Pop-Location
    }
}

# Validate test environment
function Test-TestEnvironment {
    Write-Log "Validating test environment..."
    
    # Validate test framework
    try {
        python -c "import pytest; print('pytest available')"
        Write-Log "pytest validation passed" "SUCCESS"
    }
    catch {
        Write-Log "pytest not available, but continuing..." "WARN"
    }
    
    Write-Log "Test environment validation completed" "SUCCESS"
}

# Main function
function Main {
    Write-Log "Starting HarborAI test environment setup..."
    Write-Log "Project root: $ProjectRoot"
    Write-Log "Test directory: $TestsDir"
    
    try {
        # Check basic environment
        Test-PythonEnvironment
        
        # Create directory structure
        New-TestDirectories
        
        # Install dependencies
        Install-PythonDependencies
        
        # Set environment variables
        Set-TestEnvironmentVariables
        
        # Start Docker environment
        Start-DockerTestEnvironment
        
        # Validate environment
        Test-TestEnvironment
        
        $ElapsedTime = (Get-Date) - $StartTime
        Write-Log "Test environment setup completed! Time elapsed: $($ElapsedTime.TotalSeconds.ToString('F2')) seconds" "SUCCESS"
        Write-Log "Log file: $LogFile"
        
        Write-Host ""
        Write-Host "=== Test Environment Setup Completed ===" -ForegroundColor Green
        Write-Host "You can now run the following commands for testing:" -ForegroundColor Cyan
        Write-Host "  .\run_all_tests.ps1          # Run all tests" -ForegroundColor White
        Write-Host "  .\run_performance_tests.ps1  # Run performance tests" -ForegroundColor White
        Write-Host "  .\generate_reports.ps1       # Generate test reports" -ForegroundColor White
        Write-Host ""
        
    }
    catch {
        Handle-Error "Unexpected error during test environment setup: $($_.Exception.Message)"
    }
}

# Script entry point
if ($MyInvocation.InvocationName -ne '.') {
    Main
}