#Requires -Version 7.0
# cmd_build.ps1 [full|release|minimal] [--features X,Y] [--release]
param([Parameter(ValueFromRemainingArguments)][string[]]$RawArgs)

Get-ChildItem (Join-Path $PSScriptRoot "_*.ps1") | ForEach-Object { . $_.FullName }

$features = ""
$release = $false

for ($i = 0; $i -lt $RawArgs.Count; $i++) {
    switch ($RawArgs[$i]) {
        "full"       { $features = "full" }
        "minimal"    { $features = "" }
        "release"    { $release = $true }
        "--release"  { $release = $true }
        "--features" { $i++; if ($i -lt $RawArgs.Count) { $features = $RawArgs[$i] } }
        default      { if ($RawArgs[$i] -notmatch "^-") { $features = $RawArgs[$i] } }
    }
}

$label = $features ? $features : "no features"
$mode = $release ? "release" : "dev"

Write-CmdHeader "build" "[$label] [$mode]"

$cargoArgs = @("build", "--lib")
if ($features) { $cargoArgs += "--features"; $cargoArgs += $features }
if ($release) { $cargoArgs += "--release" }

$result = Invoke-CargoWithProgress `
    -Label "build $label" `
    -CargoArgs $cargoArgs `
    -ShowProgress $true

if ($result.Success) {
    if ($result.Warnings.Count -gt 0) {
        Write-Host ""
        Write-Host "    $($result.Warnings.Count) warning(s):" -ForegroundColor Yellow
        $result.Warnings | Select-Object -First 10 | ForEach-Object {
            if ($_ -match "warning:\s*(.+)") {
                Write-Host "      $($Matches[1])" -ForegroundColor DarkYellow
            }
        }
    }

    # Binary size
    $targetDir = $release ? "target\release" : "target\debug"
    $rlib = Get-ChildItem -Path $targetDir -Filter "libignis*.rlib" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($rlib) {
        $sizeKB = [math]::Round($rlib.Length / 1024, 1)
        Write-Host "    output: $($rlib.Name) (${sizeKB} KiB)" -ForegroundColor DarkGray
    }
} else {
    $parsedErrors = Format-CargoError $result.Output
    if ($parsedErrors.Count -gt 0) {
        Write-Host ""
        foreach ($err in ($parsedErrors | Select-Object -First 5)) {
            Write-Host "    $($err.Header)" -ForegroundColor Red
            if ($err.Location) { Write-Host "      at $($err.Location)" -ForegroundColor DarkGray }
            $err.Context | Select-Object -First 5 | ForEach-Object {
                Write-Host "      $_" -ForegroundColor DarkRed
            }
            Write-Host ""
        }
    }
}