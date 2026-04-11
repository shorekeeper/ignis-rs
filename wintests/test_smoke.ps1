#Requires -Version 5.1
# test_smoke.ps1 - Run the GPU smoke test.

$ErrorActionPreference = "Continue"

Write-Host "    Smoke test (requires Vulkan GPU)" -ForegroundColor Cyan
Write-Host ""

# Check for Vulkan
$hasVulkan = $false
try {
    $vkInfo = vulkaninfo --summary 2>&1 | Out-String
    if ($vkInfo -match "deviceName") { $hasVulkan = $true }
} catch {}

if (-not $hasVulkan) {
    Write-Host "      No Vulkan ICD detected, skipping smoke test" -ForegroundColor Yellow
    Write-Host "      Install a Vulkan driver or GPU to enable this test" -ForegroundColor DarkGray
    Write-Host ""
    return [PSCustomObject]@{ Passed = 0; Failed = 0; Skipped = 1 }
}

# Show GPU info
$deviceLine = ($vkInfo -split "`n") | Where-Object { $_ -match "deviceName" } | Select-Object -First 1
$apiLine = ($vkInfo -split "`n") | Where-Object { $_ -match "apiVersion" } | Select-Object -First 1
Write-Host "      GPU: $($deviceLine.Trim())" -ForegroundColor DarkGray
if ($apiLine) {
    Write-Host "      API: $($apiLine.Trim())" -ForegroundColor DarkGray
}
Write-Host ""

Write-Host -NoNewline "      Running cargo run --example smoke_test --features full ... "
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$output = cargo run --example smoke_test --features full 2>&1
$exitCode = $LASTEXITCODE
$sw.Stop()

Write-Host ""

# Parse results from smoke test output
$resultLine = ($output | Out-String) -split "`n" | Where-Object { $_ -match "RESULTS\s+passed:" } | Select-Object -Last 1
$allOk = ($output | Out-String) -match "ALL TESTS OK"

# Show full output indented
$output | ForEach-Object {
    $line = $_
    $color = "Gray"
    if ($line -match "PASSED") { $color = "Green" }
    elseif ($line -match "FAILED|FATAL|panicked") { $color = "Red" }
    elseif ($line -match "SKIPPED") { $color = "Yellow" }
    elseif ($line -match "error\[IGN-") { $color = "DarkRed" }
    elseif ($line -match "warning\[IGN-") { $color = "DarkYellow" }
    Write-Host "      $line" -ForegroundColor $color
}

Write-Host ""
$elapsed = [math]::Round($sw.Elapsed.TotalSeconds, 1)

if ($allOk -and $exitCode -eq 0) {
    Write-Host "      Smoke test PASSED (${elapsed}s)" -ForegroundColor Green
    # Parse step counts
    $passedSteps = 0
    $skippedSteps = 0
    if ($resultLine -match "passed:\s*(\d+)\s+skipped:\s*(\d+)") {
        $passedSteps = [int]$matches[1]
        $skippedSteps = [int]$matches[2]
    }
    Write-Host ""
    return [PSCustomObject]@{ Passed = $passedSteps; Failed = 0; Skipped = $skippedSteps }
} else {
    Write-Host "      Smoke test FAILED (${elapsed}s, exit code $exitCode)" -ForegroundColor Red
    Write-Host ""
    return [PSCustomObject]@{ Passed = 0; Failed = 1; Skipped = 0 }
}