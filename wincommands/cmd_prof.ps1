# cmd_prof.ps1 [build|test] [--features X]
param([Parameter(ValueFromRemainingArguments)][string[]]$Args)


$target = if ($Args.Count -gt 0 -and $Args[0] -notmatch "^-") { $Args[0] } else { "build" }

Write-CmdHeader "prof" "[$target] timing profiler"

switch ($target) {
    "build" {
        $configs = @(
            @{ Label = "no features";     Features = "" },
            @{ Label = "tracking";        Features = "tracking" },
            @{ Label = "debug-tools";     Features = "debug-tools" },
            @{ Label = "slab-allocator";  Features = "slab-allocator" },
            @{ Label = "full";            Features = "full" }
        )

        # Clean first for accurate timing
        Write-Host "    Cleaning for fresh build times..." -ForegroundColor DarkGray
        cargo clean 2>&1 | Out-Null

        $results = @()

        foreach ($cfg in $configs) {
            Write-Host -NoNewline "    $($cfg.Label.PadRight(20)) "

            $cargoArgs = @("check", "--lib")
            if ($cfg.Features) { $cargoArgs += "--features"; $cargoArgs += $cfg.Features }

            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            & cargo @cargoArgs 2>&1 | Out-Null
            $sw.Stop()

            $ms = $sw.Elapsed.TotalMilliseconds
            $bar = "#" * [math]::Min(40, [math]::Max(1, [math]::Round($ms / 200)))
            $color = if ($ms -gt 10000) { "Red" } elseif ($ms -gt 5000) { "Yellow" } else { "Green" }

            Write-Host "[$bar] " -NoNewline -ForegroundColor $color
            Write-Host "$(Format-Duration $ms)" -ForegroundColor White

            $results += [PSCustomObject]@{ Label = $cfg.Label; Ms = $ms }

            # Clean between measurements for accurate incremental comparison
            cargo clean 2>&1 | Out-Null
        }

        if ($results.Count -ge 2) {
            $baseline = $results[0].Ms
            $full = ($results | Where-Object { $_.Label -eq "full" }).Ms
            if ($full -and $baseline) {
                $overhead = [math]::Round(($full - $baseline) / 1000, 1)
                Write-Host ""
                Write-Host "    Feature overhead: +${overhead}s from no-features to full" -ForegroundColor DarkGray
            }
        }
    }

    "test" {
        Write-Host "    Timing each test suite..." -ForegroundColor DarkGray
        Write-Host ""

        $suites = @("features", "lint", "unit", "audit", "doc", "smoke")
        $testDir = Join-Path $PSScriptRoot "..\wintests"

        foreach ($suite in $suites) {
            $script = Join-Path $testDir "test_$suite.ps1"
            if (-not (Test-Path $script)) { continue }

            Write-Host -NoNewline "    $($suite.PadRight(12)) "

            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            & $script 2>&1 | Out-Null
            $sw.Stop()

            $ms = $sw.Elapsed.TotalMilliseconds
            $bar = "#" * [math]::Min(40, [math]::Max(1, [math]::Round($ms / 500)))
            $color = if ($LASTEXITCODE -ne 0) { "Red" } elseif ($ms -gt 30000) { "Yellow" } else { "Green" }

            Write-Host "[$bar] " -NoNewline -ForegroundColor $color
            Write-Host "$(Format-Duration $ms)" -ForegroundColor White
        }
    }

    default {
        Write-Host "    Unknown: $target (use build, test)" -ForegroundColor Red
    }
}
