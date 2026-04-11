# cmd_run.ps1 [example_name] [--features X] [--release]
param([Parameter(ValueFromRemainingArguments)][string[]]$Args)


$example = "smoke_test"
$features = "full"
$release = $false

for ($i = 0; $i -lt $Args.Count; $i++) {
    switch ($Args[$i]) {
        "--features" { $i++; if ($i -lt $Args.Count) { $features = $Args[$i] } }
        "--release"  { $release = $true }
        default      { if ($Args[$i] -notmatch "^-") { $example = $Args[$i] } }
    }
}

Write-CmdHeader "run" "$example [features=$features]$(if ($release) { " release" })"

$cargoArgs = @("run", "--example", $example)
if ($features) { $cargoArgs += "--features"; $cargoArgs += $features }
if ($release) { $cargoArgs += "--release" }

$sw = [System.Diagnostics.Stopwatch]::StartNew()
& cargo @cargoArgs 2>&1 | ForEach-Object {
    $line = "$_"
    if ($line -match "error|FATAL|panicked") { Write-Host "    $line" -ForegroundColor Red }
    elseif ($line -match "^warning") { Write-Host "    $line" -ForegroundColor Yellow }
    elseif ($line -match "PASSED") { Write-Host "    $line" -ForegroundColor Green }
    else { Write-Host "    $line" }
}
$sw.Stop()

Write-Host ""
Write-Host "    Exit: $LASTEXITCODE | $(Format-Duration $sw.Elapsed.TotalMilliseconds)" -ForegroundColor $(if ($LASTEXITCODE -eq 0) { "Green" } else { "Red" })
