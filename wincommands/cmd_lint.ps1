# cmd_lint.ps1 [clippy|fmt|doc|all] [--fix]
param([Parameter(ValueFromRemainingArguments)][string[]]$Args)
Get-ChildItem (Join-Path $PSScriptRoot "_*.ps1") | ForEach-Object { . $_.FullName }

$target = "all"
$fix = $false

foreach ($a in $Args) {
    switch ($a) {
        "--fix" { $fix = $true }
        default { if ($a -notmatch "^-") { $target = $a } }
    }
}

Write-CmdHeader "lint" "[$target]$(if ($fix) { " --fix" })"

function Run-Fmt {
    if ($fix) {
        Write-Host -NoNewline "    fmt (fix) ... "
        cargo fmt --all 2>&1 | Out-Null
    } else {
        Write-Host -NoNewline "    fmt (check) ... "
        $output = cargo fmt --all -- --check 2>&1
    }
    if ($LASTEXITCODE -eq 0) {
        Write-Host "OK" -ForegroundColor Green
        return $true
    } else {
        Write-Host "FAIL" -ForegroundColor Red
        if (-not $fix) {
            $changed = @($output | Where-Object { "$_" -match "Diff in" })
            if ($changed) {
                $changed | Select-Object -First 10 | ForEach-Object {
                    Write-Host "      $_" -ForegroundColor DarkRed
                }
                Write-Host "      run 'lint fmt --fix' to auto-format" -ForegroundColor Yellow
            }
        }
        return $false
    }
}

function Run-Clippy {
    $clippyArgs = @(
        "clippy", "--all-targets", "--features", "full", "--"
        "-W", "clippy::all", "-W", "clippy::pedantic"
        "-A", "clippy::module_name_repetitions"
        "-A", "clippy::too_many_arguments"
        "-A", "clippy::missing_errors_doc"
        "-A", "clippy::must_use_candidate"
        "-A", "clippy::return_self_not_must_use"
        "-A", "clippy::cast_possible_truncation"
        "-A", "clippy::cast_sign_loss"
        "-A", "clippy::cast_precision_loss"
        "-A", "clippy::missing_panics_doc"
    )

    $result = Invoke-CargoWithProgress `
        -Label "clippy full" `
        -CargoArgs $clippyArgs `
        -ShowProgress $true

    if ($result.Success) {
        if ($result.Warnings.Count -gt 0) {
            Write-Host "    $($result.Warnings.Count) warning(s):" -ForegroundColor Yellow
            $result.Warnings | Select-Object -First 5 | ForEach-Object {
                $msg = $_ -replace "^warning:\s*", ""
                Write-Host "      $msg" -ForegroundColor DarkYellow
            }
        }
        return $true
    } else {
        $result.Errors | Select-Object -First 10 | ForEach-Object {
            Write-Host "      $_" -ForegroundColor DarkRed
        }
        return $false
    }
}

function Run-Doc {
    Write-Host -NoNewline "    doc warnings ... "
    $output = cargo doc --features full --no-deps 2>&1
    $docErrors = @($output | Where-Object { "$_" -match "^error" })
    $docWarnings = @($output | Where-Object { "$_" -match "^warning" })

    if ($docErrors.Count -gt 0) {
        Write-Host "FAIL ($($docErrors.Count) error(s))" -ForegroundColor Red
        $docErrors | Select-Object -First 5 | ForEach-Object { Write-Host "      $_" -ForegroundColor DarkRed }
        return $false
    } elseif ($docWarnings.Count -gt 0) {
        Write-Host "OK ($($docWarnings.Count) warning(s))" -ForegroundColor Yellow
        return $true
    } else {
        Write-Host "OK (clean)" -ForegroundColor Green
        return $true
    }
}

switch ($target) {
    "fmt"    { Run-Fmt }
    "clippy" { Run-Clippy }
    "doc"    { Run-Doc }
    "all"    { Run-Fmt; Run-Clippy; Run-Doc }
    default  { Write-Host "    Unknown: $target (use fmt, clippy, doc, all)" -ForegroundColor Red }
}
