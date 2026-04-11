#Requires -Version 5.1
# test_size.ps1 - Binary size analysis.

$ErrorActionPreference = "Continue"

Write-Host "    Binary size analysis" -ForegroundColor Cyan
Write-Host ""

$configs = @(
    @{ Label = "no features (dev)";  Args = "--lib";                     Release = $false },
    @{ Label = "full (dev)";         Args = "--lib --features full";     Release = $false },
    @{ Label = "no features (rel)";  Args = "--lib --release";           Release = $true },
    @{ Label = "full (rel)";         Args = "--lib --release --features full"; Release = $true }
)

$results = @()

foreach ($cfg in $configs) {
    Write-Host -NoNewline "      Building [$($cfg.Label)] ... "
    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    $buildArgs = $cfg.Args -split " "
    $output = & cargo build @buildArgs 2>&1
    $exitCode = $LASTEXITCODE
    $sw.Stop()

    if ($exitCode -ne 0) {
        Write-Host "BUILD FAILED" -ForegroundColor Red
        continue
    }

    # Find the rlib
    $targetDir = if ($cfg.Release) { "target\release" } else { "target\debug" }
    $rlib = Get-ChildItem -Path $targetDir -Filter "libignis*.rlib" -ErrorAction SilentlyContinue | Select-Object -First 1

    if ($rlib) {
        $sizeKB = [math]::Round($rlib.Length / 1024, 1)
        $sizeMB = [math]::Round($rlib.Length / (1024 * 1024), 2)
        $display = if ($sizeMB -ge 1) { "${sizeMB} MiB" } else { "${sizeKB} KiB" }
        Write-Host "$display " -NoNewline -ForegroundColor White
        Write-Host "($([math]::Round($sw.Elapsed.TotalSeconds, 1))s)" -ForegroundColor DarkGray

        $results += [PSCustomObject]@{
            Config = $cfg.Label
            Bytes  = $rlib.Length
            Display = $display
        }
    } else {
        Write-Host "rlib not found" -ForegroundColor Yellow
    }
}

# Show delta between no-features and full
if ($results.Count -ge 2) {
    Write-Host ""
    Write-Host "    Size comparison:" -ForegroundColor DarkGray
    foreach ($r in $results) {
        $bar = "#" * [math]::Min(50, [math]::Max(1, [math]::Round($r.Bytes / 1024 / 20)))
        Write-Host "      $($r.Display.PadLeft(12))  [$bar] $($r.Config)" -ForegroundColor Gray
    }

    $noFeatDev = $results | Where-Object { $_.Config -eq "no features (dev)" }
    $fullDev = $results | Where-Object { $_.Config -eq "full (dev)" }
    if ($noFeatDev -and $fullDev) {
        $delta = $fullDev.Bytes - $noFeatDev.Bytes
        $deltaKB = [math]::Round($delta / 1024, 1)
        Write-Host ""
        Write-Host "      debug-tools overhead: +${deltaKB} KiB" -ForegroundColor DarkGray
    }
}

Write-Host ""

return [PSCustomObject]@{
    Passed  = $results.Count
    Failed  = $configs.Count - $results.Count
    Skipped = 0
}