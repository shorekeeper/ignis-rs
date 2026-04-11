# cmd_test.ps1 [all|unit|smoke|features|lint|audit|doc|size|miri] [--step N] [--filter X]
param([Parameter(ValueFromRemainingArguments)][string[]]$Args)
Get-ChildItem (Join-Path $PSScriptRoot "_*.ps1") | ForEach-Object { . $_.FullName }

$suite = "all"
$stepFilter = $null
$testFilter = $null

for ($i = 0; $i -lt $Args.Count; $i++) {
    switch ($Args[$i]) {
        "--step"   { $i++; if ($i -lt $Args.Count) { $stepFilter = [int]$Args[$i] } }
        "--filter" { $i++; if ($i -lt $Args.Count) { $testFilter = $Args[$i] } }
        default    { if ($Args[$i] -notmatch "^-") { $suite = $Args[$i] } }
    }
}

Write-CmdHeader "test" "[$suite]$(if ($stepFilter) { " step=$stepFilter" })$(if ($testFilter) { " filter=$testFilter" })"

$testScript = Join-Path $PSScriptRoot "..\wintests"

switch ($suite) {
    "all" {
        Write-Host "    Running full CI test suite..." -ForegroundColor DarkGray
        Write-Host ""
        $phases = @("features", "lint", "unit", "audit", "doc", "smoke", "size")
        $total = 0; $failed = 0
        foreach ($phase in $phases) {
            $script = Join-Path $testScript "test_$phase.ps1"
            if (Test-Path $script) {
                Write-Host "    [$phase]" -ForegroundColor Cyan
                $result = & $script
                if ($result -is [PSCustomObject] -and $null -ne $result.Failed) {
                    $total += $result.Passed + $result.Failed
                    $failed += $result.Failed
                }
            }
        }
        Write-Host ""
        Write-Host "    Total: $($total - $failed)/$total passed" -ForegroundColor $(if ($failed -gt 0) { "Red" } else { "Green" })
    }

    "unit" {
        $cargoArgs = @("test", "--lib", "--features", "full")
        if ($testFilter) { $cargoArgs += "--"; $cargoArgs += $testFilter }

        $result = Invoke-CargoWithProgress `
            -Label "test unit" `
            -CargoArgs $cargoArgs `
            -ShowProgress $true `
            -ShowOutput $true  # Stream test results in real-time

        Write-Host ""
        $resultLine = $result.Output | Where-Object { $_ -match "test result:" } | Select-Object -Last 1
        if ($resultLine) {
            $color = $result.Success ? "Green" : "Red"
            Write-Host "    $resultLine" -ForegroundColor $color
        }

        if (-not $result.Success) {
            $testFails = Parse-TestOutput $result.Output
            if ($testFails.Count -gt 0) {
                Write-Host ""
                Write-Host "    Failures:" -ForegroundColor Red
                foreach ($tf in $testFails) {
                    Write-Host "      $($tf.TestName)" -ForegroundColor Red
                    if ($tf.PanicMessage) {
                        Write-Host "        $($tf.PanicMessage)" -ForegroundColor DarkRed
                    }
                }
            }
        }
    }

    "smoke" {
        $cargoArgs = @("run", "--example", "smoke_test", "--features", "full")

        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $output = & cargo @cargoArgs 2>&1
        $exitCode = $LASTEXITCODE
        $sw.Stop()

        foreach ($line in $output) {
            $str = "$line"

            # Step filter: dim non-matching steps
            if ($stepFilter -and $str -match "^\[(\d+)/") {
                $num = [int]$Matches[1]
                if ($num -ne $stepFilter) {
                    # Show but dimmed
                    if ($str -match "PASSED|SKIPPED") {
                        Write-Host "    $str" -ForegroundColor DarkGray
                    }
                    continue
                }
            }

            # Colorize
            if ($str -match "PASSED") { Write-Host "    $str" -ForegroundColor Green }
            elseif ($str -match "SKIPPED") { Write-Host "    $str" -ForegroundColor Yellow }
            elseif ($str -match "FATAL|panicked") { Write-Host "    $str" -ForegroundColor Red }
            elseif ($str -match "^\[") { Write-Host "    $str" -ForegroundColor White }
            elseif ($str -match "error\[IGN-") { Write-Host "    $str" -ForegroundColor DarkRed }
            elseif ($str -match "warning\[IGN-") { Write-Host "    $str" -ForegroundColor DarkYellow }
            elseif ($str -match "RESULTS|ALL TESTS") { Write-Host "    $str" -ForegroundColor Cyan }
            else { Write-Host "    $str" -ForegroundColor Gray }
        }
    }

    default {
        # Try to find matching wintests script
        $script = Join-Path $testScript "test_$suite.ps1"
        if (Test-Path $script) {
            & $script
        } else {
            Write-Host "    Unknown test suite: $suite" -ForegroundColor Red
            Write-Host "    Available: all, unit, smoke, features, lint, audit, doc, size, miri" -ForegroundColor DarkGray
        }
    }
}
