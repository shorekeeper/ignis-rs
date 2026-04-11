#Requires -Version 5.1
# test_miri.ps1 - Miri UB detection on pure-Rust logic.
# Requires: rustup +nightly component add miri

$ErrorActionPreference = "Continue"

Write-Host "    Miri undefined behavior detection" -ForegroundColor Cyan
Write-Host "    (Miri cannot execute Vulkan calls, only pure-Rust logic)" -ForegroundColor DarkGray
Write-Host ""

# Check if nightly + miri available
$nightlyCheck = rustup run nightly rustc --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "      Nightly toolchain not installed, skipping" -ForegroundColor Yellow
    Write-Host "      Install with: rustup toolchain install nightly" -ForegroundColor DarkGray
    Write-Host ""
    return [PSCustomObject]@{ Passed = 0; Failed = 0; Skipped = 1 }
}

$miriCheck = cargo +nightly miri --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "      Miri not installed, skipping" -ForegroundColor Yellow
    Write-Host "      Install with: rustup +nightly component add miri" -ForegroundColor DarkGray
    Write-Host ""
    return [PSCustomObject]@{ Passed = 0; Failed = 0; Skipped = 1 }
}

Write-Host "      Miri version: $($miriCheck | Select-Object -First 1)" -ForegroundColor DarkGray
Write-Host ""

Write-Host -NoNewline "      cargo +nightly miri test --lib --features full ... "
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$output = cargo +nightly miri test --features full --lib 2>&1
$exitCode = $LASTEXITCODE
$sw.Stop()

$elapsed = [math]::Round($sw.Elapsed.TotalSeconds, 1)

$ubLines = @($output | Where-Object { $_ -match "Undefined Behavior|error" })
$resultLine = $output | Where-Object { $_ -match "test result:" } | Select-Object -Last 1

if ($exitCode -eq 0 -and $ubLines.Count -eq 0) {
    Write-Host "OK (${elapsed}s)" -ForegroundColor Green
    if ($resultLine) {
        Write-Host "        $resultLine" -ForegroundColor DarkGray
    }

    return [PSCustomObject]@{ Passed = 1; Failed = 0; Skipped = 0 }
} else {
    Write-Host "ISSUES FOUND (${elapsed}s)" -ForegroundColor Red

    $output | ForEach-Object {
        $color = if ($_ -match "Undefined Behavior|error") { "Red" } else { "Gray" }
        Write-Host "        $_" -ForegroundColor $color
    }

    Write-Host ""
    return [PSCustomObject]@{ Passed = 0; Failed = 1; Skipped = 0 }
}