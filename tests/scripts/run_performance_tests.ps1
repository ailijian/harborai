<#
.SYNOPSIS
    HarborAI Performance Test Runner Script

.DESCRIPTION
    This script runs performance tests for the HarborAI project, including:
    - Load testing
    - Stress testing
    - Memory usage testing
    - Response time testing
    - Throughput testing
    - Resource utilization monitoring
    - Performance benchmarking

.PARAMETER TestType
    Type of performance tests to run (load, stress, memory, response, throughput, all)

.PARAMETER Duration
    Test duration in seconds (default: 300)

.PARAMETER Concurrency
    Number of concurrent users/requests (default: 10)

.PARAMETER RampUp
    Ramp-up time in seconds (default: 30)

.PARAMETER Endpoint
    API endpoint to test (default: http://localhost:8000)

.PARAMETER ReportFormat
    Performance report format (html, json, csv)

.PARAMETER Verbose
    Show verbose output

.PARAMETER Help
    Show help information

.EXAMPLE
    .\run_performance_tests.ps1
    Run all performance tests with default settings

.EXAMPLE
    .\run_performance_tests.ps1 -TestType load -Duration 600 -Concurrency 50
    Run load tests for 10 minutes with 50 concurrent users

.EXAMPLE
    .\run_performance_tests.ps1 -TestType stress -Endpoint http://localhost:8080 -ReportFormat json
    Run stress tests against custom endpoint with JSON reports
#>

param(
    [ValidateSet("load", "stress", "memory", "response", "throughput", "all")]
    [string]$TestType = "all",
    [int]$Duration = 300,
    [int]$Concurrency = 10,
    [int]$RampUp = 30,
    [string]$Endpoint = "http://localhost:8000",
    [ValidateSet("html", "json", "csv")]
    [string]$ReportFormat = "html",
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
$PerformanceDir = Join-Path $TestsDir "performance"
$ReportsDir = Join-Path $TestsDir "reports" "performance"
$LogFile = Join-Path $TestsDir "logs" "performance_test_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

# Create necessary directories
$DirsToCreate = @($PerformanceDir, $ReportsDir, (Split-Path -Parent $LogFile))
foreach ($Dir in $DirsToCreate) {
    if (-not (Test-Path $Dir)) {
        New-Item -Path $Dir -ItemType Directory -Force | Out-Null
    }
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
    Write-Log "Performance test execution failed, please check error information" "ERROR"
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

# Check performance test environment
function Test-PerformanceEnvironment {
    Write-Log "Checking performance test environment..."
    
    # Check Python
    if (-not (Test-Command "python")) {
        Handle-Error "Python not found, please run setup_test_env.ps1 first"
    }
    
    # Check required Python packages
    $RequiredPackages = @("locust", "psutil", "requests", "matplotlib")
    foreach ($Package in $RequiredPackages) {
        try {
            python -c "import $Package" 2>$null
            Write-Log "$Package is available" "SUCCESS"
        }
        catch {
            Write-Log "$Package not found, installing..." "WARN"
            try {
                python -m pip install $Package
                Write-Log "$Package installed successfully" "SUCCESS"
            }
            catch {
                Handle-Error "Failed to install $Package"
            }
        }
    }
    
    # Test endpoint connectivity
    try {
        $Response = Invoke-WebRequest -Uri $Endpoint -Method GET -TimeoutSec 10 -ErrorAction Stop
        Write-Log "Endpoint $Endpoint is accessible (Status: $($Response.StatusCode))" "SUCCESS"
    }
    catch {
        Write-Log "Warning: Endpoint $Endpoint is not accessible. Some tests may fail." "WARN"
    }
    
    Write-Log "Performance test environment check completed" "SUCCESS"
}

# Create Locust test file
function New-LocustTestFile {
    $LocustFile = Join-Path $PerformanceDir "locustfile.py"
    
    $LocustContent = @"
from locust import HttpUser, task, between
import json
import random

class HarborAIUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """Called when a user starts"""
        pass
    
    @task(3)
    def test_health_check(self):
        """Test health check endpoint"""
        self.client.get("/health")
    
    @task(2)
    def test_api_status(self):
        """Test API status endpoint"""
        self.client.get("/api/status")
    
    @task(1)
    def test_api_info(self):
        """Test API info endpoint"""
        self.client.get("/api/info")
    
    @task(1)
    def test_post_data(self):
        """Test POST request with data"""
        data = {
            "test_id": random.randint(1, 1000),
            "message": f"Performance test message {random.randint(1, 100)}"
        }
        self.client.post("/api/test", json=data)
"@
    
    Set-Content -Path $LocustFile -Value $LocustContent -Encoding UTF8
    Write-Log "Locust test file created: $LocustFile" "SUCCESS"
    return $LocustFile
}

# Run load testing
function Invoke-LoadTest {
    Write-Log "Running load test..."
    
    $LocustFile = New-LocustTestFile
    $ReportFile = Join-Path $ReportsDir "load_test_report.html"
    $StatsFile = Join-Path $ReportsDir "load_test_stats.csv"
    
    $LocustArgs = @(
        "-f", $LocustFile,
        "--host=$Endpoint",
        "--users=$Concurrency",
        "--spawn-rate=1",
        "--run-time=${Duration}s",
        "--headless",
        "--html=$ReportFile",
        "--csv=$($StatsFile.Replace('.csv', ''))"
    )
    
    try {
        Push-Location $PerformanceDir
        python -m locust @LocustArgs
        Write-Log "Load test completed successfully" "SUCCESS"
        Write-Log "Report saved to: $ReportFile"
    }
    catch {
        Write-Log "Load test failed: $($_.Exception.Message)" "ERROR"
    }
    finally {
        Pop-Location
    }
}

# Run stress testing
function Invoke-StressTest {
    Write-Log "Running stress test..."
    
    $LocustFile = New-LocustTestFile
    $ReportFile = Join-Path $ReportsDir "stress_test_report.html"
    $StatsFile = Join-Path $ReportsDir "stress_test_stats.csv"
    
    # Stress test with higher concurrency
    $StressConcurrency = $Concurrency * 3
    
    $LocustArgs = @(
        "-f", $LocustFile,
        "--host=$Endpoint",
        "--users=$StressConcurrency",
        "--spawn-rate=5",
        "--run-time=${Duration}s",
        "--headless",
        "--html=$ReportFile",
        "--csv=$($StatsFile.Replace('.csv', ''))"
    )
    
    try {
        Push-Location $PerformanceDir
        python -m locust @LocustArgs
        Write-Log "Stress test completed successfully" "SUCCESS"
        Write-Log "Report saved to: $ReportFile"
    }
    catch {
        Write-Log "Stress test failed: $($_.Exception.Message)" "ERROR"
    }
    finally {
        Pop-Location
    }
}

# Run memory usage testing
function Invoke-MemoryTest {
    Write-Log "Running memory usage test..."
    
    $MemoryTestScript = Join-Path $PerformanceDir "memory_test.py"
    
    $MemoryTestContent = @"
import psutil
import time
import json
import requests
from datetime import datetime

def monitor_memory(duration, endpoint):
    """Monitor memory usage during API calls"""
    start_time = time.time()
    memory_data = []
    
    while time.time() - start_time < duration:
        # Get current memory usage
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        
        # Make API call
        try:
            response = requests.get(f"{endpoint}/health", timeout=5)
            response_time = response.elapsed.total_seconds()
        except Exception as e:
            response_time = -1
        
        memory_data.append({
            'timestamp': datetime.now().isoformat(),
            'memory_percent': memory_info.percent,
            'memory_available': memory_info.available,
            'memory_used': memory_info.used,
            'cpu_percent': cpu_percent,
            'response_time': response_time
        })
        
        time.sleep(1)
    
    return memory_data

if __name__ == "__main__":
    endpoint = "$Endpoint"
    duration = $Duration
    
    print(f"Starting memory monitoring for {duration} seconds...")
    data = monitor_memory(duration, endpoint)
    
    # Save results
    with open("$($ReportsDir.Replace('\', '/'))/memory_test_results.json", 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Memory test completed. Results saved.")
"@
    
    Set-Content -Path $MemoryTestScript -Value $MemoryTestContent -Encoding UTF8
    
    try {
        Push-Location $PerformanceDir
        python $MemoryTestScript
        Write-Log "Memory test completed successfully" "SUCCESS"
    }
    catch {
        Write-Log "Memory test failed: $($_.Exception.Message)" "ERROR"
    }
    finally {
        Pop-Location
    }
}

# Run response time testing
function Invoke-ResponseTimeTest {
    Write-Log "Running response time test..."
    
    $ResponseTestScript = Join-Path $PerformanceDir "response_time_test.py"
    
    $ResponseTestContent = @"
import requests
import time
import json
import statistics
from datetime import datetime

def test_response_times(endpoint, num_requests=100):
    """Test API response times"""
    response_times = []
    successful_requests = 0
    failed_requests = 0
    
    print(f"Testing response times with {num_requests} requests...")
    
    for i in range(num_requests):
        try:
            start_time = time.time()
            response = requests.get(f"{endpoint}/health", timeout=10)
            end_time = time.time()
            
            response_time = (end_time - start_time) * 1000  # Convert to milliseconds
            response_times.append(response_time)
            successful_requests += 1
            
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{num_requests} requests")
                
        except Exception as e:
            failed_requests += 1
            print(f"Request {i + 1} failed: {e}")
    
    if response_times:
        results = {
            'timestamp': datetime.now().isoformat(),
            'endpoint': endpoint,
            'total_requests': num_requests,
            'successful_requests': successful_requests,
            'failed_requests': failed_requests,
            'min_response_time': min(response_times),
            'max_response_time': max(response_times),
            'avg_response_time': statistics.mean(response_times),
            'median_response_time': statistics.median(response_times),
            'p95_response_time': statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else max(response_times),
            'p99_response_time': statistics.quantiles(response_times, n=100)[98] if len(response_times) > 100 else max(response_times)
        }
    else:
        results = {
            'timestamp': datetime.now().isoformat(),
            'endpoint': endpoint,
            'total_requests': num_requests,
            'successful_requests': 0,
            'failed_requests': failed_requests,
            'error': 'No successful requests'
        }
    
    return results

if __name__ == "__main__":
    endpoint = "$Endpoint"
    results = test_response_times(endpoint)
    
    # Save results
    with open("$($ReportsDir.Replace('\', '/'))/response_time_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Response time test completed. Results saved.")
    print(f"Average response time: {results.get('avg_response_time', 'N/A')} ms")
"@
    
    Set-Content -Path $ResponseTestScript -Value $ResponseTestContent -Encoding UTF8
    
    try {
        Push-Location $PerformanceDir
        python $ResponseTestScript
        Write-Log "Response time test completed successfully" "SUCCESS"
    }
    catch {
        Write-Log "Response time test failed: $($_.Exception.Message)" "ERROR"
    }
    finally {
        Pop-Location
    }
}

# Run throughput testing
function Invoke-ThroughputTest {
    Write-Log "Running throughput test..."
    
    $LocustFile = New-LocustTestFile
    $ReportFile = Join-Path $ReportsDir "throughput_test_report.html"
    $StatsFile = Join-Path $ReportsDir "throughput_test_stats.csv"
    
    $LocustArgs = @(
        "-f", $LocustFile,
        "--host=$Endpoint",
        "--users=$Concurrency",
        "--spawn-rate=2",
        "--run-time=${Duration}s",
        "--headless",
        "--html=$ReportFile",
        "--csv=$($StatsFile.Replace('.csv', ''))"
    )
    
    try {
        Push-Location $PerformanceDir
        python -m locust @LocustArgs
        Write-Log "Throughput test completed successfully" "SUCCESS"
        Write-Log "Report saved to: $ReportFile"
    }
    catch {
        Write-Log "Throughput test failed: $($_.Exception.Message)" "ERROR"
    }
    finally {
        Pop-Location
    }
}

# Generate performance summary report
function New-PerformanceSummary {
    Write-Log "Generating performance summary report..."
    
    $SummaryFile = Join-Path $ReportsDir "performance_summary.html"
    
    $SummaryContent = @"
<!DOCTYPE html>
<html>
<head>
    <title>HarborAI Performance Test Summary</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .success { color: green; }
        .warning { color: orange; }
        .error { color: red; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>HarborAI Performance Test Summary</h1>
        <p><strong>Test Date:</strong> $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')</p>
        <p><strong>Test Type:</strong> $TestType</p>
        <p><strong>Endpoint:</strong> $Endpoint</p>
        <p><strong>Duration:</strong> $Duration seconds</p>
        <p><strong>Concurrency:</strong> $Concurrency users</p>
    </div>
    
    <div class="section">
        <h2>Test Configuration</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Test Type</td><td>$TestType</td></tr>
            <tr><td>Duration</td><td>$Duration seconds</td></tr>
            <tr><td>Concurrency</td><td>$Concurrency users</td></tr>
            <tr><td>Ramp-up Time</td><td>$RampUp seconds</td></tr>
            <tr><td>Target Endpoint</td><td>$Endpoint</td></tr>
            <tr><td>Report Format</td><td>$ReportFormat</td></tr>
        </table>
    </div>
    
    <div class="section">
        <h2>Test Results</h2>
        <p>Detailed test results are available in the following files:</p>
        <ul>
            <li><a href="load_test_report.html">Load Test Report</a></li>
            <li><a href="stress_test_report.html">Stress Test Report</a></li>
            <li><a href="throughput_test_report.html">Throughput Test Report</a></li>
            <li><a href="memory_test_results.json">Memory Test Results</a></li>
            <li><a href="response_time_results.json">Response Time Results</a></li>
        </ul>
    </div>
    
    <div class="section">
        <h2>Log Files</h2>
        <p>Test execution logs: <code>$LogFile</code></p>
    </div>
</body>
</html>
"@
    
    Set-Content -Path $SummaryFile -Value $SummaryContent -Encoding UTF8
    Write-Log "Performance summary report generated: $SummaryFile" "SUCCESS"
}

# Main function
function Main {
    Write-Log "Starting HarborAI performance test execution..."
    Write-Log "Test type: $TestType"
    Write-Log "Duration: $Duration seconds"
    Write-Log "Concurrency: $Concurrency users"
    Write-Log "Endpoint: $Endpoint"
    Write-Log "Report format: $ReportFormat"
    Write-Log "Reports directory: $ReportsDir"
    
    try {
        # Check performance test environment
        Test-PerformanceEnvironment
        
        # Run tests based on type
        switch ($TestType) {
            "load" {
                Invoke-LoadTest
            }
            "stress" {
                Invoke-StressTest
            }
            "memory" {
                Invoke-MemoryTest
            }
            "response" {
                Invoke-ResponseTimeTest
            }
            "throughput" {
                Invoke-ThroughputTest
            }
            "all" {
                Invoke-LoadTest
                Invoke-StressTest
                Invoke-MemoryTest
                Invoke-ResponseTimeTest
                Invoke-ThroughputTest
            }
        }
        
        # Generate summary report
        New-PerformanceSummary
        
        $ElapsedTime = (Get-Date) - $StartTime
        Write-Log "Performance test execution completed! Time elapsed: $($ElapsedTime.TotalSeconds.ToString('F2')) seconds" "SUCCESS"
        Write-Log "Log file: $LogFile"
        Write-Log "Reports directory: $ReportsDir"
        
        Write-Host ""
        Write-Host "=== Performance Test Execution Completed ===" -ForegroundColor Green
        Write-Host "Test type: $TestType" -ForegroundColor Cyan
        Write-Host "Duration: $Duration seconds" -ForegroundColor Cyan
        Write-Host "Concurrency: $Concurrency users" -ForegroundColor Cyan
        Write-Host "Reports directory: $ReportsDir" -ForegroundColor Cyan
        Write-Host ""
        
    }
    catch {
        Handle-Error "Unexpected error during performance test execution: $($_.Exception.Message)"
    }
}

# Script entry point
if ($MyInvocation.InvocationName -ne '.') {
    Main
}