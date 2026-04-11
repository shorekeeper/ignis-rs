#Requires -Version 5.1
# test_features.ps1 - Tests every feature combination compiles.
# Returns a result object with Passed/Failed/Skipped counts.

$ErrorActionPreference = "Stop"

Write-Host "    Feature matrix compilation test" -ForegroundColor Cyan
Write-Host "    Verifying all feature combinations produce valid code" -ForegroundColor DarkGray
Write-Host ""

$features = @(
    "",
    "tracking",
    "descriptors",
    "debug-tools",
    "slab-allocator",
    "swapchain",
    "interop",
    "tracking,descriptors",
    "tracking,debug-tools",
    "tracking,slab-allocator",
    "descriptors,debug-tools",
    "swapchain,interop",
    "tracking,descriptors,debug-tools",
    "tracking,descriptors,debug-tools,slab-allocator",
    "full"
)

$passed = 0
$failed = 0
$failedNames = @()
$timings = @()

foreach ($f in $features) {
    $label = if ($f -eq "") { "(no features)" } else { $f }
    Write-Host -NoNewline "      [$($passed + $failed + 1)/$($features.Count)] $label ... "

    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    if ($f -eq "") {
        $output = cargo check --lib 2>&1
    } else {
        $output = cargo check --lib --features $f 2>&1
    }
    $exitCode = $LASTEXITCODE
    $sw.Stop()

    $elapsed = $sw.Elapsed.TotalSeconds

    if ($exitCode -eq 0) {
        Write-Host "OK " -NoNewline -ForegroundColor Green
        Write-Host "($([math]::Round($elapsed, 1))s)" -ForegroundColor DarkGray
        $passed++
    } else {
        Write-Host "FAIL " -NoNewline -ForegroundColor Red
        Write-Host "($([math]::Round($elapsed, 1))s)" -ForegroundColor DarkGray
        $failed++
        $failedNames += $label

        # Show first 30 lines of compiler output for diagnosis
        Write-Host ""
        $output | Select-Object -First 30 | ForEach-Object {
            Write-Host "        $_" -ForegroundColor DarkRed
        }
        $totalLines = ($output | Measure-Object -Line).Lines
        if ($totalLines -gt 30) {
            Write-Host "        ... ($($totalLines - 30) more lines)" -ForegroundColor DarkGray
        }
        Write-Host ""
    }

    $timings += [PSCustomObject]@{ Feature = $label; Time = $elapsed; Ok = ($exitCode -eq 0) }
}

# Summary
Write-Host ""
Write-Host "    Results: $passed passed, $failed failed out of $($features.Count)" -ForegroundColor $(if ($failed -gt 0) { "Red" } else { "Green" })

if ($failed -gt 0) {
    Write-Host "    Failed combinations:" -ForegroundColor Red
    foreach ($f in $failedNames) {
        Write-Host "      - $f" -ForegroundColor Red
    }
}

# Timing breakdown
$slowest = $timings | Sort-Object Time -Descending | Select-Object -First 3
if ($slowest) {
    Write-Host ""
    Write-Host "    Slowest compilations:" -ForegroundColor DarkGray
    foreach ($t in $slowest) {
        $marker = if ($t.Ok) { "OK" } else { "FAIL" }
        $color = if ($t.Ok) { "DarkGray" } else { "Red" }
        Write-Host "      $([math]::Round($t.Time, 1))s  $marker  $($t.Feature)" -ForegroundColor $color
    }
}

Write-Host ""

return [PSCustomObject]@{
    Passed  = $passed
    Failed  = $failed
    Skipped = 0
}