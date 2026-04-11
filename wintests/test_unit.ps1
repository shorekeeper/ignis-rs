#Requires -Version 5.1
# test_unit.ps1 - Run library unit tests with detailed output.

$ErrorActionPreference = "Continue"

Write-Host "    Unit test suite" -ForegroundColor Cyan
Write-Host ""

$passed = 0
$failed = 0

# ── Unit tests (full features) ───────────────────────────────────────────────

Write-Host -NoNewline "      [1/2] cargo test --lib --features full ... "
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$testOutput = cargo test --lib --features full 2>&1
$exitCode = $LASTEXITCODE
$sw.Stop()

# Parse test output for counts
$resultLine = $testOutput | Where-Object { $_ -match "test result:" } | Select-Object -Last 1

if ($exitCode -eq 0) {
    Write-Host "OK " -NoNewline -ForegroundColor Green
    Write-Host "($([math]::Round($sw.Elapsed.TotalSeconds, 1))s)" -ForegroundColor DarkGray
    if ($resultLine) {
        Write-Host "        $resultLine" -ForegroundColor DarkGray
    }
    $passed++
} else {
    Write-Host "FAIL " -NoNewline -ForegroundColor Red
    Write-Host "($([math]::Round($sw.Elapsed.TotalSeconds, 1))s)" -ForegroundColor DarkGray
    $failed++

    # Show failed test names
    $failedTests = $testOutput | Where-Object { $_ -match "^test .+ \.\.\. FAILED" }
    if ($failedTests) {
        Write-Host ""
        Write-Host "        Failed tests:" -ForegroundColor Red
        $failedTests | ForEach-Object { Write-Host "          $_" -ForegroundColor DarkRed }
    }

    # Show the assertion/panic output
    $testOutput | Where-Object { $_ -match "panicked|assertion|thread.*panicked" } | ForEach-Object {
        Write-Host "        $_" -ForegroundColor DarkRed
    }
    if ($resultLine) {
        Write-Host "        $resultLine" -ForegroundColor Red
    }
}

# ── Unit tests (no features) ─────────────────────────────────────────────────

Write-Host -NoNewline "      [2/2] cargo test --lib (no features) ... "
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$testOutput = cargo test --lib 2>&1
$exitCode = $LASTEXITCODE
$sw.Stop()

$resultLine = $testOutput | Where-Object { $_ -match "test result:" } | Select-Object -Last 1

if ($exitCode -eq 0) {
    Write-Host "OK " -NoNewline -ForegroundColor Green
    Write-Host "($([math]::Round($sw.Elapsed.TotalSeconds, 1))s)" -ForegroundColor DarkGray
    if ($resultLine) {
        Write-Host "        $resultLine" -ForegroundColor DarkGray
    }
    $passed++
} else {
    Write-Host "FAIL " -NoNewline -ForegroundColor Red
    Write-Host "($([math]::Round($sw.Elapsed.TotalSeconds, 1))s)" -ForegroundColor DarkGray
    $failed++

    $failedTests = $testOutput | Where-Object { $_ -match "^test .+ \.\.\. FAILED" }
    if ($failedTests) {
        Write-Host ""
        $failedTests | ForEach-Object { Write-Host "          $_" -ForegroundColor DarkRed }
    }
}

Write-Host ""
Write-Host "    Unit tests: $passed passed, $failed failed" -ForegroundColor $(if ($failed -gt 0) { "Red" } else { "Green" })
Write-Host ""

return [PSCustomObject]@{
    Passed  = $passed
    Failed  = $failed
    Skipped = 0
}