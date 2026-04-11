#Requires -Version 7.0
# cmd_check.ps1 [full|minimal] [--features X,Y]
param([Parameter(ValueFromRemainingArguments)][string[]]$RawArgs)

Get-ChildItem (Join-Path $PSScriptRoot "_*.ps1") | ForEach-Object { . $_.FullName }

$features = ""
for ($i = 0; $i -lt $RawArgs.Count; $i++) {
    switch ($RawArgs[$i]) {
        "full"       { $features = "full" }
        "minimal"    { $features = "" }
        "--features" { $i++; if ($i -lt $RawArgs.Count) { $features = $RawArgs[$i] } }
        default      { if ($RawArgs[$i] -notmatch "^-") { $features = $RawArgs[$i] } }
    }
}

$label = $features ? $features : "no features"
Write-CmdHeader "check" "[$label]"

$cargoArgs = @("check", "--lib")
if ($features) { $cargoArgs += "--features"; $cargoArgs += $features }

$result = Invoke-CargoWithProgress `
    -Label "check $label" `
    -CargoArgs $cargoArgs `
    -ShowProgress $true

if ($result.Warnings.Count -gt 0) {
    $uniqueWarns = @{}
    foreach ($w in $result.Warnings) {
        $msg = "$w" -replace "warning:\s*", "" -replace "\s*$", ""
        if ($msg.Length -gt 80) { $msg = $msg.Substring(0, 77) + "..." }
        $uniqueWarns[$msg] = ($uniqueWarns[$msg] ?? 0) + 1
    }
    Write-Host ""
    foreach ($kv in ($uniqueWarns.GetEnumerator() | Sort-Object Value -Descending | Select-Object -First 10)) {
        $count = $kv.Value -gt 1 ? " (x$($kv.Value))" : ""
        Write-Host "      $($kv.Key)$count" -ForegroundColor DarkYellow
    }
}

if (-not $result.Success) {
    $parsedErrors = Format-CargoError $result.Output
    foreach ($err in ($parsedErrors | Select-Object -First 5)) {
        Write-Host ""
        Write-Host "    $($err.Header)" -ForegroundColor Red
        if ($err.Location) { Write-Host "      at $($err.Location)" -ForegroundColor DarkGray }
        $err.Context | Select-Object -First 4 | ForEach-Object {
            Write-Host "      $_" -ForegroundColor DarkRed
        }
    }
}