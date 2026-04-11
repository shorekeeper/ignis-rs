# cmd_lint.ps1 [clippy|fmt|doc|all] [--fix]
param([Parameter(ValueFromRemainingArguments)][string[]]$Args)


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
    $mode = if ($fix) { "--fix --allow-dirty --allow-staged" } else { "" }
    Write-Host -NoNewline "    clippy (full) ... "

    $clippyArgs = @("clippy", "--all-targets", "--features", "full")
    if ($fix) { $clippyArgs += "--fix"; $clippyArgs += "--allow-dirty"; $clippyArgs += "--allow-staged" }
    $clippyArgs += "--"
    $clippyArgs += "-W"; $clippyArgs += "clippy::all"
    $clippyArgs += "-W"; $clippyArgs += "clippy::pedantic"
    $clippyArgs += "-A"; $clippyArgs += "clippy::module_name_repetitions"
    $clippyArgs += "-A"; $clippyArgs += "clippy::too_many_arguments"
    $clippyArgs += "-A"; $clippyArgs += "clippy::missing_errors_doc"
    $clippyArgs += "-A"; $clippyArgs += "clippy::must_use_candidate"
    $clippyArgs += "-A"; $clippyArgs += "clippy::return_self_not_must_use"
    $clippyArgs += "-A"; $clippyArgs += "clippy::cast_possible_truncation"
    $clippyArgs += "-A"; $clippyArgs += "clippy::cast_sign_loss"
    $clippyArgs += "-A"; $clippyArgs += "clippy::cast_precision_loss"
    $clippyArgs += "-A"; $clippyArgs += "clippy::missing_panics_doc"

    $output = & cargo @clippyArgs 2>&1
    $exitCode = $LASTEXITCODE

    $warnings = @($output | Where-Object { "$_" -match "^warning\[" -or "$_" -match "^warning:" })

    if ($exitCode -eq 0) {
        if ($warnings.Count -gt 0) {
            Write-Host "OK ($($warnings.Count) warning(s))" -ForegroundColor Yellow
            $warnings | Select-Object -First 5 | ForEach-Object {
                $msg = "$_" -replace "^warning:\s*", ""
                Write-Host "      $msg" -ForegroundColor DarkYellow
            }
        } else {
            Write-Host "OK (clean)" -ForegroundColor Green
        }
        return $true
    } else {
        Write-Host "FAIL" -ForegroundColor Red
        $output | Where-Object { "$_" -match "^error" } | Select-Object -First 10 | ForEach-Object {
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
