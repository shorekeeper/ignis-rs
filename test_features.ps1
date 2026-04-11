# test_features.ps1
# Tests every feature combination compiles.

$ErrorActionPreference = "Stop"

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
    "tracking,descriptors,debug-tools",
    "tracking,descriptors,debug-tools,slab-allocator",
    "full"
)

$failed = @()
$passed = 0

foreach ($f in $features) {
    $label = if ($f -eq "") { "(no features)" } else { $f }
    Write-Host -NoNewline "  Building [$label]... "

    if ($f -eq "") {
        $result = cargo build --lib 2>&1
    } else {
        $result = cargo build --lib --features $f 2>&1
    }

    if ($LASTEXITCODE -eq 0) {
        Write-Host "OK" -ForegroundColor Green
        $passed++
    } else {
        Write-Host "FAIL" -ForegroundColor Red
        $failed += $label
        $result | Select-Object -First 20 | ForEach-Object {
            Write-Host "    $_" -ForegroundColor DarkRed
        }
    }
}

Write-Host ""
Write-Host "Results: $passed passed, $($failed.Count) failed out of $($features.Count)" -ForegroundColor Cyan

if ($failed.Count -gt 0) {
    Write-Host "Failed:" -ForegroundColor Red
    foreach ($f in $failed) {
        Write-Host "  - $f" -ForegroundColor Red
    }
    exit 1
}