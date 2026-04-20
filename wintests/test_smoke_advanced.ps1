#Requires -Version 5.1
# test_smoke_advanced.ps1 - Run the advanced features smoke test.

$ErrorActionPreference = "Continue"

Write-Host "  Advanced features smoke test" -ForegroundColor Cyan
Write-Host ""

# Check for Vulkan
$hasVulkan = $false
try {
    $vkInfo = vulkaninfo --summary 2>&1 | Out-String
    if ($vkInfo -match "deviceName") { $hasVulkan = $true }
} catch {}

if (-not $hasVulkan) {
    Write-Host "  No Vulkan ICD detected, skipping advanced smoke test" -ForegroundColor Yellow
    Write-Host ""
    return [PSCustomObject]@{ Passed = 0; Failed = 0; Skipped = 1 }
}

Write-Host -NoNewline "  Running cargo run --example smoke_test_advanced --features full ... "
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$output = cargo run --example smoke_test_advanced --features full 2>&1
$exitCode = $LASTEXITCODE
$sw.Stop()
Write-Host ""

$allOk = ($output | Out-String) -match "ALL TESTS OK"

$output | ForEach-Object {
    $line = $_
    $color = "Gray"
    if ($line -match "PASSED") { $color = "Green" }
    elseif ($line -match "FAILED|FATAL|panicked") { $color = "Red" }
    elseif ($line -match "SKIPPED") { $color = "Yellow" }
    elseif ($line -match "^\[") { $color = "White" }
    elseif ($line -match "RESULTS|ALL TESTS") { $color = "Cyan" }
    Write-Host "    $line" -ForegroundColor $color
}

Write-Host ""
$elapsed = [math]::Round($sw.Elapsed.TotalSeconds, 1)
if ($allOk -and $exitCode -eq 0) {
    $resultLine = ($output | Out-String) -split "`n" | Where-Object { $_ -match "RESULTS\s+passed:" } | Select-Object -Last 1
    $passedSteps = 0
    $skippedSteps = 0
    if ($resultLine -match "passed:\s*(\d+)\s+skipped:\s*(\d+)") {
        $passedSteps = [int]$matches[1]
        $skippedSteps = [int]$matches[2]
    }
    Write-Host "  Advanced smoke test PASSED (${elapsed}s)" -ForegroundColor Green
    Write-Host ""
    return [PSCustomObject]@{ Passed = $passedSteps; Failed = 0; Skipped = $skippedSteps }
} else {
    Write-Host "  Advanced smoke test FAILED (${elapsed}s, exit $exitCode)" -ForegroundColor Red
    Write-Host ""
    return [PSCustomObject]@{ Passed = 0; Failed = 1; Skipped = 0 }
}