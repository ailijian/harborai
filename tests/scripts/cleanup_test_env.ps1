<#
.SYNOPSIS
    HarborAI Test Environment Cleanup Script

.DESCRIPTION
    This script cleans up the HarborAI test environment, including:
    - Stopping and removing Docker containers
    - Cleaning up test databases
    - Removing temporary files and logs
    - Cleaning up test reports
    - Resetting environment variables
    - Cleaning up Python virtual environments
    - Removing test data

.PARAMETER CleanupType
    Type of cleanup to perform (containers, database, files, reports, env, python, all)

.PARAMETER Force
    Force cleanup without confirmation prompts

.PARAMETER KeepLogs
    Keep log files during cleanup

.PARAMETER KeepReports
    Keep test reports during cleanup

.PARAMETER Verbose
    Show verbose output

.PARAMETER Help
    Show help information

.EXAMPLE
    .\cleanup_test_env.ps1
    Perform complete cleanup with confirmation prompts

.EXAMPLE
    .\cleanup_test_env.ps1 -CleanupType containers -Force
    Force cleanup of Docker containers only

.EXAMPLE
    .\cleanup_test_env.ps1 -CleanupType all -Force -KeepReports
    Force complete cleanup but keep test reports
#>

param(
    [ValidateSet("containers", "database", "files", "reports", "env", "python", "all")]
    [string]$CleanupType = "all",
    [switch]$Force,
    [switch]$KeepLogs,
    [switch]$KeepReports,
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
$LogFile = Join-Path $TestsDir "logs" "cleanup_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

# Create log directory if it doesn't exist
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
    Write-Log "Cleanup failed, please check error information" "ERROR"
    exit 1
}

# Confirmation function
function Confirm-Action {
    param(
        [string]$Message,
        [string]$Title = "Confirm Action"
    )
    
    if ($Force) {
        return $true
    }
    
    $Response = Read-Host "$Message (y/N)"
    return ($Response -eq 'y' -or $Response -eq 'Y' -or $Response -eq 'yes' -or $Response -eq 'Yes')
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

# Stop and remove Docker containers
function Stop-DockerContainers {
    Write-Log "Cleaning up Docker containers..."
    
    if (-not (Test-Command "docker")) {
        Write-Log "Docker not found, skipping container cleanup" "WARN"
        return
    }
    
    try {
        # Check if Docker is running
        docker version | Out-Null
        
        # Find HarborAI test containers
        $Containers = docker ps -a --filter "name=harborai" --format "{{.Names}}" 2>$null
        
        if ($Containers) {
            Write-Log "Found HarborAI containers: $($Containers -join ', ')"
            
            if (Confirm-Action "Stop and remove HarborAI Docker containers?") {
                foreach ($Container in $Containers) {
                    Write-Log "Stopping container: $Container"
                    docker stop $Container 2>$null | Out-Null
                    
                    Write-Log "Removing container: $Container"
                    docker rm $Container 2>$null | Out-Null
                }
                Write-Log "Docker containers cleaned up successfully" "SUCCESS"
            } else {
                Write-Log "Docker container cleanup skipped by user"
            }
        } else {
            Write-Log "No HarborAI containers found"
        }
        
        # Clean up test networks
        $Networks = docker network ls --filter "name=harborai" --format "{{.Name}}" 2>$null
        if ($Networks) {
            Write-Log "Found HarborAI networks: $($Networks -join ', ')"
            
            if (Confirm-Action "Remove HarborAI Docker networks?") {
                foreach ($Network in $Networks) {
                    Write-Log "Removing network: $Network"
                    docker network rm $Network 2>$null | Out-Null
                }
                Write-Log "Docker networks cleaned up successfully" "SUCCESS"
            }
        }
        
        # Clean up test volumes
        $Volumes = docker volume ls --filter "name=harborai" --format "{{.Name}}" 2>$null
        if ($Volumes) {
            Write-Log "Found HarborAI volumes: $($Volumes -join ', ')"
            
            if (Confirm-Action "Remove HarborAI Docker volumes?") {
                foreach ($Volume in $Volumes) {
                    Write-Log "Removing volume: $Volume"
                    docker volume rm $Volume 2>$null | Out-Null
                }
                Write-Log "Docker volumes cleaned up successfully" "SUCCESS"
            }
        }
        
    }
    catch {
        Write-Log "Error during Docker cleanup: $($_.Exception.Message)" "ERROR"
    }
}

# Clean up test databases
function Clear-TestDatabases {
    Write-Log "Cleaning up test databases..."
    
    try {
        # Check if PostgreSQL client is available
        if (Test-Command "psql") {
            if (Confirm-Action "Drop test databases?") {
                # This would connect to PostgreSQL and drop test databases
                # For safety, we'll just log what would be done
                Write-Log "Would drop test databases: harborai_test, harborai_test_integration" "WARN"
                Write-Log "Database cleanup is disabled for safety - implement as needed"
            }
        } else {
            Write-Log "PostgreSQL client not found, skipping database cleanup" "WARN"
        }
        
        # Clean up SQLite test databases
        $SqliteFiles = Get-ChildItem -Path $TestsDir -Filter "*.db" -Recurse -ErrorAction SilentlyContinue
        if ($SqliteFiles) {
            Write-Log "Found SQLite test databases: $($SqliteFiles.Count) files"
            
            if (Confirm-Action "Remove SQLite test database files?") {
                foreach ($File in $SqliteFiles) {
                    Write-Log "Removing SQLite database: $($File.FullName)"
                    Remove-Item -Path $File.FullName -Force
                }
                Write-Log "SQLite databases cleaned up successfully" "SUCCESS"
            }
        }
        
    }
    catch {
        Write-Log "Error during database cleanup: $($_.Exception.Message)" "ERROR"
    }
}

# Clean up temporary files and logs
function Clear-TemporaryFiles {
    Write-Log "Cleaning up temporary files..."
    
    try {
        # Clean up temporary directories
        $TempDirs = @(
            (Join-Path $TestsDir "temp"),
            (Join-Path $TestsDir "tmp"),
            (Join-Path $TestsDir "cache"),
            (Join-Path $TestsDir "__pycache__")
        )
        
        foreach ($TempDir in $TempDirs) {
            if (Test-Path $TempDir) {
                Write-Log "Found temporary directory: $TempDir"
                
                if (Confirm-Action "Remove temporary directory: $TempDir?") {
                    Remove-Item -Path $TempDir -Recurse -Force
                    Write-Log "Removed temporary directory: $TempDir" "SUCCESS"
                }
            }
        }
        
        # Clean up Python cache files
        $PycacheFiles = Get-ChildItem -Path $ProjectRoot -Name "__pycache__" -Recurse -Directory -ErrorAction SilentlyContinue
        if ($PycacheFiles) {
            Write-Log "Found Python cache directories: $($PycacheFiles.Count) directories"
            
            if (Confirm-Action "Remove Python cache directories?") {
                foreach ($CacheDir in $PycacheFiles) {
                    $FullPath = Join-Path $ProjectRoot $CacheDir
                    Write-Log "Removing Python cache: $FullPath"
                    Remove-Item -Path $FullPath -Recurse -Force -ErrorAction SilentlyContinue
                }
                Write-Log "Python cache directories cleaned up successfully" "SUCCESS"
            }
        }
        
        # Clean up .pyc files
        $PycFiles = Get-ChildItem -Path $ProjectRoot -Filter "*.pyc" -Recurse -ErrorAction SilentlyContinue
        if ($PycFiles) {
            Write-Log "Found .pyc files: $($PycFiles.Count) files"
            
            if (Confirm-Action "Remove .pyc files?") {
                foreach ($PycFile in $PycFiles) {
                    Remove-Item -Path $PycFile.FullName -Force
                }
                Write-Log ".pyc files cleaned up successfully" "SUCCESS"
            }
        }
        
        # Clean up log files (if not keeping them)
        if (-not $KeepLogs) {
            $LogsDir = Join-Path $TestsDir "logs"
            if (Test-Path $LogsDir) {
                $LogFiles = Get-ChildItem -Path $LogsDir -Filter "*.log" -ErrorAction SilentlyContinue | Where-Object { $_.FullName -ne $LogFile }
                
                if ($LogFiles) {
                    Write-Log "Found log files: $($LogFiles.Count) files"
                    
                    if (Confirm-Action "Remove old log files?") {
                        foreach ($LogFileItem in $LogFiles) {
                            Remove-Item -Path $LogFileItem.FullName -Force
                        }
                        Write-Log "Log files cleaned up successfully" "SUCCESS"
                    }
                }
            }
        } else {
            Write-Log "Keeping log files as requested"
        }
        
    }
    catch {
        Write-Log "Error during temporary files cleanup: $($_.Exception.Message)" "ERROR"
    }
}

# Clean up test reports
function Clear-TestReports {
    if ($KeepReports) {
        Write-Log "Keeping test reports as requested"
        return
    }
    
    Write-Log "Cleaning up test reports..."
    
    try {
        $ReportsDir = Join-Path $TestsDir "reports"
        
        if (Test-Path $ReportsDir) {
            $ReportFiles = Get-ChildItem -Path $ReportsDir -ErrorAction SilentlyContinue
            
            if ($ReportFiles) {
                Write-Log "Found test reports: $($ReportFiles.Count) files/directories"
                
                if (Confirm-Action "Remove test reports?") {
                    Remove-Item -Path $ReportsDir -Recurse -Force
                    Write-Log "Test reports cleaned up successfully" "SUCCESS"
                }
            } else {
                Write-Log "No test reports found"
            }
        } else {
            Write-Log "Reports directory not found"
        }
        
    }
    catch {
        Write-Log "Error during test reports cleanup: $($_.Exception.Message)" "ERROR"
    }
}

# Reset environment variables
function Reset-EnvironmentVariables {
    Write-Log "Resetting test environment variables..."
    
    try {
        # List of test environment variables to remove
        $TestEnvVars = @(
            "HARBORAI_TEST_MODE",
            "HARBORAI_TEST_DB_HOST",
            "HARBORAI_TEST_DB_PORT",
            "HARBORAI_TEST_DB_NAME",
            "HARBORAI_TEST_DB_USER",
            "HARBORAI_TEST_DB_PASSWORD",
            "HARBORAI_TEST_API_URL",
            "HARBORAI_TEST_LOG_LEVEL",
            "PYTHONPATH"
        )
        
        $RemovedVars = @()
        foreach ($VarName in $TestEnvVars) {
            if ([Environment]::GetEnvironmentVariable($VarName, "Process")) {
                [Environment]::SetEnvironmentVariable($VarName, $null, "Process")
                $RemovedVars += $VarName
            }
        }
        
        if ($RemovedVars.Count -gt 0) {
            Write-Log "Removed environment variables: $($RemovedVars -join ', ')" "SUCCESS"
        } else {
            Write-Log "No test environment variables found to remove"
        }
        
        # Remove test environment file
        $EnvFile = Join-Path $TestsDir ".env.test"
        if (Test-Path $EnvFile) {
            if (Confirm-Action "Remove test environment file: $EnvFile?") {
                Remove-Item -Path $EnvFile -Force
                Write-Log "Test environment file removed: $EnvFile" "SUCCESS"
            }
        }
        
    }
    catch {
        Write-Log "Error during environment variables cleanup: $($_.Exception.Message)" "ERROR"
    }
}

# Clean up Python virtual environments
function Clear-PythonEnvironments {
    Write-Log "Cleaning up Python virtual environments..."
    
    try {
        # Look for virtual environment directories
        $VenvDirs = @(
            (Join-Path $ProjectRoot "venv"),
            (Join-Path $ProjectRoot ".venv"),
            (Join-Path $TestsDir "venv"),
            (Join-Path $TestsDir ".venv")
        )
        
        foreach ($VenvDir in $VenvDirs) {
            if (Test-Path $VenvDir) {
                Write-Log "Found virtual environment: $VenvDir"
                
                if (Confirm-Action "Remove virtual environment: $VenvDir?") {
                    Remove-Item -Path $VenvDir -Recurse -Force
                    Write-Log "Virtual environment removed: $VenvDir" "SUCCESS"
                }
            }
        }
        
        # Clean up pip cache
        if (Test-Command "python") {
            try {
                $PipCacheDir = python -m pip cache dir 2>$null
                if ($PipCacheDir -and (Test-Path $PipCacheDir)) {
                    Write-Log "Found pip cache directory: $PipCacheDir"
                    
                    if (Confirm-Action "Clear pip cache?") {
                        python -m pip cache purge 2>$null
                        Write-Log "Pip cache cleared successfully" "SUCCESS"
                    }
                }
            }
            catch {
                Write-Log "Could not access pip cache" "WARN"
            }
        }
        
    }
    catch {
        Write-Log "Error during Python environments cleanup: $($_.Exception.Message)" "ERROR"
    }
}

# Display cleanup summary
function Show-CleanupSummary {
    param([datetime]$StartTime)
    
    $ElapsedTime = (Get-Date) - $StartTime
    
    Write-Host ""
    Write-Host "=== Cleanup Summary ===" -ForegroundColor Green
    Write-Host "Cleanup type: $CleanupType" -ForegroundColor Cyan
    Write-Host "Force mode: $Force" -ForegroundColor Cyan
    Write-Host "Keep logs: $KeepLogs" -ForegroundColor Cyan
    Write-Host "Keep reports: $KeepReports" -ForegroundColor Cyan
    Write-Host "Time elapsed: $($ElapsedTime.TotalSeconds.ToString('F2')) seconds" -ForegroundColor Cyan
    Write-Host "Log file: $LogFile" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Cleanup completed successfully!" -ForegroundColor Green
    Write-Host ""
}

# Main function
function Main {
    Write-Log "Starting HarborAI test environment cleanup..."
    Write-Log "Cleanup type: $CleanupType"
    Write-Log "Force mode: $Force"
    Write-Log "Keep logs: $KeepLogs"
    Write-Log "Keep reports: $KeepReports"
    
    if (-not $Force) {
        Write-Host ""
        Write-Host "=== HarborAI Test Environment Cleanup ===" -ForegroundColor Yellow
        Write-Host "This will clean up the test environment based on your settings." -ForegroundColor Yellow
        Write-Host "Cleanup type: $CleanupType" -ForegroundColor Cyan
        Write-Host "Keep logs: $KeepLogs" -ForegroundColor Cyan
        Write-Host "Keep reports: $KeepReports" -ForegroundColor Cyan
        Write-Host ""
        
        if (-not (Confirm-Action "Do you want to proceed with the cleanup?")) {
            Write-Log "Cleanup cancelled by user"
            exit 0
        }
    }
    
    try {
        # Perform cleanup based on type
        switch ($CleanupType) {
            "containers" {
                Stop-DockerContainers
            }
            "database" {
                Clear-TestDatabases
            }
            "files" {
                Clear-TemporaryFiles
            }
            "reports" {
                Clear-TestReports
            }
            "env" {
                Reset-EnvironmentVariables
            }
            "python" {
                Clear-PythonEnvironments
            }
            "all" {
                Stop-DockerContainers
                Clear-TestDatabases
                Clear-TemporaryFiles
                Clear-TestReports
                Reset-EnvironmentVariables
                Clear-PythonEnvironments
            }
        }
        
        Write-Log "Cleanup completed successfully!" "SUCCESS"
        Show-CleanupSummary -StartTime $StartTime
        
    }
    catch {
        Handle-Error "Unexpected error during cleanup: $($_.Exception.Message)"
    }
}

# Script entry point
if ($MyInvocation.InvocationName -ne '.') {
    Main
}