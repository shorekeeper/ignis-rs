#Requires -Version 5.1
# test_lint.ps1 - Clippy, rustfmt, and doc warning checks.

$ErrorActionPreference = "Continue"

Write-Host "    Lint suite" -ForegroundColor Cyan
Write-Host ""

$passed = 0
$failed = 0

# rustfmt

Write-Host -NoNewline "      [1/4] cargo fmt --check ... "
$fmtOutput = cargo fmt --all -- --check 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "OK" -ForegroundColor Green
    $passed++
} else {
    Write-Host "FAIL" -ForegroundColor Red
    $failed++
    $fmtOutput | ForEach-Object { Write-Host "        $_" -ForegroundColor DarkRed }
    Write-Host ""
    Write-Host "        Run 'cargo fmt' to fix formatting" -ForegroundColor Yellow
}

# cargo check

Write-Host -NoNewline "      [2/4] cargo check --features full ... "
$checkOutput = cargo check --features full 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "OK" -ForegroundColor Green
    $passed++
} else {
    Write-Host "FAIL" -ForegroundColor Red
    $failed++
    $checkOutput | Select-Object -First 30 | ForEach-Object {
        Write-Host "        $_" -ForegroundColor DarkRed
    }
}

# clippy (full features) 

Write-Host -NoNewline "      [3/4] cargo clippy --features full ... "
$clippyOutput = cargo clippy --all-targets --features full -- `
    -W clippy::all -W clippy::pedantic `
    -A clippy::module_name_repetitions `
    -A clippy::too_many_arguments `
    -A clippy::missing_errors_doc `
    -A clippy::must_use_candidate `
    -A clippy::return_self_not_must_use `
    -A clippy::cast_possible_truncation `
    -A clippy::cast_sign_loss `
    -A clippy::cast_precision_loss `
    -A clippy::missing_panics_doc 2>&1

if ($LASTEXITCODE -eq 0) {
    # Count warnings even on success
    $warnings = ($clippyOutput | Where-Object { $_ -match "^warning" }).Count
    if ($warnings -gt 0) {
        Write-Host "OK ($warnings warning(s))" -ForegroundColor Yellow
    } else {
        Write-Host "OK (clean)" -ForegroundColor Green
    }
    $passed++
} else {
    Write-Host "FAIL" -ForegroundColor Red
    $failed++
    $clippyOutput | Where-Object { $_ -match "warning|error" } | Select-Object -First 40 | ForEach-Object {
        Write-Host "        $_" -ForegroundColor DarkRed
    }
}

# doc warnings

Write-Host -NoNewline "      [4/4] cargo doc --features full ... "
$docOutput = cargo doc --features full --no-deps 2>&1
$docWarnings = ($docOutput | Where-Object { $_ -match "warning" })
$docErrors = ($docOutput | Where-Object { $_ -match "error" })

if ($docErrors.Count -gt 0) {
    Write-Host "FAIL ($($docErrors.Count) error(s))" -ForegroundColor Red
    $failed++
    $docErrors | ForEach-Object { Write-Host "        $_" -ForegroundColor DarkRed }
} elseif ($docWarnings.Count -gt 0) {
    Write-Host "OK ($($docWarnings.Count) warning(s))" -ForegroundColor Yellow
    $passed++
    $docWarnings | Select-Object -First 10 | ForEach-Object {
        Write-Host "        $_" -ForegroundColor DarkYellow
    }
    if ($docWarnings.Count -gt 10) {
        Write-Host "        ... $($docWarnings.Count - 10) more" -ForegroundColor DarkGray
    }
} else {
    Write-Host "OK (clean)" -ForegroundColor Green
    $passed++
}

Write-Host ""
Write-Host "    Lint: $passed passed, $failed failed" -ForegroundColor $(if ($failed -gt 0) { "Red" } else { "Green" })
Write-Host ""

return [PSCustomObject]@{
    Passed  = $passed
    Failed  = $failed
    Skipped = 0
}