#Requires -Version 7.0
# cmd_prof.ps1 [build|test] [--features X]
param([Parameter(ValueFromRemainingArguments)][string[]]$RawArgs)

Get-ChildItem (Join-Path $PSScriptRoot "_*.ps1") | ForEach-Object { . $_.FullName }

$target = if ($RawArgs.Count -gt 0 -and $RawArgs[0] -notmatch "^-") { $RawArgs[0] } else { "build" }

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

        Write-Host "    Cleaning for fresh build times..." -ForegroundColor DarkGray
        cargo clean 2>&1 | Out-Null
        Write-Host ""

        $results = @()

        foreach ($cfg in $configs) {
            $cargoArgs = @("check", "--lib")
            if ($cfg.Features) { $cargoArgs += "--features"; $cargoArgs += $cfg.Features }

            $result = Invoke-CargoWithProgress `
                -Label $cfg.Label.PadRight(20) `
                -CargoArgs $cargoArgs `
                -ShowProgress $true `
                -ShowOutput $false

            $results += [PSCustomObject]@{
                Label   = $cfg.Label
                Ms      = $result.Elapsed.TotalMilliseconds
                Success = $result.Success
            }

            # Clean between measurements
            cargo clean 2>&1 | Out-Null
        }

        # Summary
        if ($results.Count -ge 2) {
            $baseline = ($results | Where-Object { $_.Label -eq "no features" }).Ms
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

        $suites = @(
            @{ Label = "features"; CargoArgs = @("check", "--lib") },
            @{ Label = "lint";     CargoArgs = @("clippy", "--all-targets", "--features", "full", "--", "-W", "clippy::all") },
            @{ Label = "unit";     CargoArgs = @("test", "--lib", "--features", "full") },
            @{ Label = "doc";      CargoArgs = @("doc", "--features", "full", "--no-deps") }
        )

        foreach ($suite in $suites) {
            $result = Invoke-CargoWithProgress `
                -Label $suite.Label.PadRight(12) `
                -CargoArgs $suite.CargoArgs `
                -ShowProgress $true `
                -ShowOutput $false
        }
    }

    default {
        Write-Host "    Unknown: $target (use build, test)" -ForegroundColor Red
    }
}