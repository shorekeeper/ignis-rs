# cmd_check.ps1 [full|minimal] [--features X,Y]
param([Parameter(ValueFromRemainingArguments)][string[]]$Args)


$features = ""
for ($i = 0; $i -lt $Args.Count; $i++) {
    switch ($Args[$i]) {
        "full"       { $features = "full" }
        "minimal"    { $features = "" }
        "--features" { $i++; if ($i -lt $Args.Count) { $features = $Args[$i] } }
        default      { if ($Args[$i] -notmatch "^-") { $features = $Args[$i] } }
    }
}

$label = if ($features) { $features } else { "no features" }
Write-CmdHeader "check" "[$label]"

$cargoArgs = @("check", "--lib")
if ($features) { $cargoArgs += "--features"; $cargoArgs += $features }

$sw = [System.Diagnostics.Stopwatch]::StartNew()
$output = & cargo @cargoArgs 2>&1
$exitCode = $LASTEXITCODE
$sw.Stop()

$errors = @($output | Where-Object { "$_" -match "^error" })
$warnings = @($output | Where-Object { "$_" -match "^warning" })

if ($exitCode -eq 0) {
    $warnStr = if ($warnings.Count -gt 0) { "$($warnings.Count) warning(s)" } else { "clean" }
    Write-SubStep "check" "OK" "($warnStr, $(Format-Duration $sw.Elapsed.TotalMilliseconds))"

    # Show unique warnings collapsed
    if ($warnings.Count -gt 0) {
        $uniqueWarns = @{}
        foreach ($w in $warnings) {
            $msg = "$w" -replace "warning:\s*", "" -replace "\s*$", ""
            if ($msg.Length -gt 80) { $msg = $msg.Substring(0, 77) + "..." }
            if (-not $uniqueWarns.ContainsKey($msg)) { $uniqueWarns[$msg] = 0 }
            $uniqueWarns[$msg]++
        }
        Write-Host ""
        foreach ($kv in ($uniqueWarns.GetEnumerator() | Sort-Object Value -Descending | Select-Object -First 10)) {
            $count = if ($kv.Value -gt 1) { " (x$($kv.Value))" } else { "" }
            Write-Host "      $($kv.Key)$count" -ForegroundColor DarkYellow
        }
    }
} else {
    Write-SubStep "check" "FAIL" "($($errors.Count) error(s))"

    $parsedErrors = Format-CargoError ($output | ForEach-Object { "$_" })
    foreach ($err in ($parsedErrors | Select-Object -First 5)) {
        Write-Host ""
        Write-Host "    $($err.Header)" -ForegroundColor Red
        if ($err.Location) { Write-Host "      at $($err.Location)" -ForegroundColor DarkGray }
        $err.Context | Select-Object -First 4 | ForEach-Object {
            Write-Host "      $_" -ForegroundColor DarkRed
        }
    }
}
